import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import os
import time
import argparse
import json
import torch
import gc
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler  # Import AMP components
from env import AttrDict, build_env
from dataset import Dataset, mag_pha_stft, mag_pha_istft, get_dataset_filelist
from models.model import MPNet, pesq_score, phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = MPNet(h).to(device)
    discriminator = MetricDiscriminator().to(device)

    # Initialize GradScaler for AMP (optional for bf16, but included for robustness)
    scaler = GradScaler(enabled=torch.cuda.is_bf16_supported())  # Enable scaler only if bf16 is supported

    if rank == 0:
        print(generator)
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        print('Total Parameters: {:.3f}M'.format(num_params/1e6))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_indexes, validation_indexes = get_dataset_filelist(a)

    trainset = Dataset(training_indexes, a.input_clean_wavs_dir, a.input_noisy_wavs_dir, h.segment_size, h.sampling_rate,
                       split=True, n_cache_reuse=0, shuffle=False if h.num_gpus > 1 else True, device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    if rank == 0:
        validset = Dataset(validation_indexes, a.input_clean_wavs_dir, a.input_noisy_wavs_dir, h.segment_size, h.sampling_rate,
                           split=False, shuffle=False, n_cache_reuse=0, device=device)
        
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()

    best_pesq = 0

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, noisy_audio = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            # Use autocast for mixed precision with bf16
            with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                # Discriminator
                optim_d.zero_grad()
                metric_r = discriminator(clean_mag, clean_mag)
                metric_g = discriminator(clean_mag, mag_g_hat.detach())
                loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
                
                if batch_pesq_score := batch_pesq(clean_audio.cpu().numpy(), audio_g.detach().cpu().numpy()):
                    loss_disc_g = F.mse_loss(torch.tensor(batch_pesq_score, device=device, dtype=torch.bfloat16), metric_g.flatten())
                else:
                    print('pesq is None!')
                    loss_disc_g = 0
                
                loss_disc_all = loss_disc_r + loss_disc_g
                # Backward pass with scaler
                scaler.scale(loss_disc_all).backward()
                scaler.step(optim_d)
                scaler.update()

                # Generator
                optim_g.zero_grad()
                # L2 Magnitude Loss
                loss_mag = F.mse_loss(clean_mag, mag_g)
                # Anti-wrapping Phase Loss
                loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
                loss_pha = loss_ip + loss_gd + loss_iaf
                # L2 Complex Loss
                loss_com = F.mse_loss(clean_com, com_g) * 2
                # L2 Consistency Loss
                loss_stft = F.mse_loss(com_g, com_g_hat) * 2
                # Time Loss
                loss_time = F.l1_loss(clean_audio, audio_g)
                # Metric Loss
                metric_g = discriminator(clean_mag, mag_g_hat)
                loss_metric = F.mse_loss(metric_g.flatten(), one_labels)

                loss_gen_all = loss_mag * 0.9 + loss_pha * 0.3 + loss_com * 0.1 + loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2

                # Backward pass with scaler
                scaler.scale(loss_gen_all).backward()
                scaler.step(optim_g)
                scaler.update()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g)
                        pha_error = (ip_error + gd_error + iaf_error).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        stft_error = F.mse_loss(com_g, com_g_hat).item()
                    print('Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric loss: {:4.3f}, Magnitude Loss : {:4.3f}, Phase Loss : {:4.3f}, Complex Loss : {:4.3f}, Time Loss : {:4.3f}, STFT Loss : {:4.3f}, s/b : {:4.3f}'.
                           format(steps, loss_gen_all, loss_disc_all, metric_error, mag_error, pha_error, com_error, time_error, stft_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)

                # Validation
                if steps % a.validation_interval == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    gc.collect()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    val_stft_err_tot = 0
                    with torch.no_grad():
                        with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                            for j, batch in enumerate(validation_loader):
                                clean_audio, noisy_audio = batch
                                clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                                noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))

                                clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                                noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                                mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                                audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                                mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                                audios_r += torch.split(clean_audio, 1, dim=0)
                                audios_g += torch.split(audio_g, 1, dim=0)

                                val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                                val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g)
                                val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                                val_com_err_tot += F.mse_loss(clean_com, com_g).item()
                                val_stft_err_tot += F.mse_loss(com_g, com_g_hat).item()

                                del clean_audio, noisy_audio, mag_g, pha_g, com_g, audio_g, mag_g_hat, pha_g_hat, com_g_hat
                                torch.cuda.empty_cache()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_stft_err = val_stft_err_tot / (j+1)
                        val_pesq_score = pesq_score(audios_r, audios_g, h).item()
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar("Validation/PESQ Score", val_pesq_score, steps)
                        sw.add_scalar("Validation/Magnitude Loss", val_mag_err, steps)
                        sw.add_scalar("Validation/Phase Loss", val_pha_err, steps)
                        sw.add_scalar("Validation/Complex Loss", val_com_err, steps)
                        sw.add_scalar("Validation/Consistency Loss", val_stft_err, steps)
                    
                    if epoch >= a.best_checkpoint_start_epoch:
                        if val_pesq_score > best_pesq:
                            best_pesq = val_pesq_score
                            best_checkpoint_path = "{}/g_best".format(a.checkpoint_path)
                            save_checkpoint(best_checkpoint_path,
                                        {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})

                    generator.train()

            steps += 1

            torch.cuda.empty_cache()

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_clean_wavs_dir', default='VoiceBank+DEMAND/wavs_clean')
    parser.add_argument('--input_noisy_wavs_dir', default='VoiceBank+DEMAND/wavs_noisy')
    parser.add_argument('--input_training_file', default='VoiceBank+DEMAND/training.txt')
    parser.add_argument('--input_validation_file', default='VoiceBank+DEMAND/test.txt')
    parser.add_argument('--checkpoint_path', default='cp_model')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=40, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
        # Check for bf16 support
        if not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 is not supported on this GPU. Falling back to float32.")
    else:
        print("CUDA not available. bfloat16 requires CUDA support.")
        return

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        while True:
            try:
                train(0, a, h)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                h.batch_size -= 1
                print(f"CUDA OOM detected. Reducing batch size to {h.batch_size} and retrying...")
                if h.batch_size == 0:
                    print("Not even a single batch fits in memory. Get a better GPU, mate. Terminating early...")
                    break

if __name__ == '__main__':
    main()