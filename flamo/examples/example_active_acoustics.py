import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from flamo import dsp, system
from flamo.functional import (
    db2mag,
    mag2db,
    get_magnitude,
    get_eigenvalues,
    # WGN_reverb,
)
from flamo.optimize.dataset import DatasetColorless_mod, load_dataset
from flamo.optimize.trainer import Trainer

torch.manual_seed(130297)

# ==================================================================================================
# ===================================== Active Acoustics Class =====================================

class AA(nn.Module):
    """
    Active Acoustics (AA) model presented at DAFx24, 3-7 September, Guilford, UK.
    Reference:
        De Bortoli G., Dal Santo G., Prawda K., Lokki T., Välimäki V., and Schlecht S. J.
        Differentiable Active Acoustics---Optimizing Stability via Gradient Descent
        Int. Conf. on Digital Audio Effects (DAFx), Sep. 2024.
    """
    def __init__(self, n_S: int, n_M: int, n_L: int, n_A: int, room_name: str, fs: int=48000, nfft: int=2**11, FIR_order: int=100, wgn_RT: float=1.0, alias_decay_db: float=0):
        r"""
        Initialize the Active Acoustics (AA) model.
        Stores system parameters, RIRs, and filters.

            **Args**:
                - n_S (int): number of natural sound sources.
                - n_M (int): number of microphones.
                - n_L (int): number of loudspeakers.
                - n_A (int): number of audience positions.
                - fs (int, optional): sampling frequency. Defaults to 48000.
                - nfft (int, optional): number of frequency bins. Defaults to 2**11.
                - FIR_order (int, optional): order of the FIR filters. Defaults to 100.
                - wgn_RT (float, optional): reverberation time of the WGN reverb. Defaults to 1.0.
                - alias_decay_db (float, optional): Time-alias decay in dB. Defaults to 0.
        """
        nn.Module.__init__(self)

        # Processing resolution
        self.fs = fs
        self.nfft = nfft

        # Sources, transducers, and audience
        self.n_S = n_S
        self.n_M = n_M
        self.n_L = n_L
        self.n_A = n_A

        # Physical room
        self.__Room = AA_RIRs(dir=room_name, n_S=self.n_S, n_L=self.n_L, n_M=self.n_M, n_A=self.n_A, fs=self.fs)
        self.H_SM = dsp.Filter(size=(self.__Room.RIR_length, n_M, n_S), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_SM.assign_value(self.__Room.get_scs_to_mcs())
        self.H_SA = dsp.Filter(size=(self.__Room.RIR_length, n_A, n_S), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_SA.assign_value(self.__Room.get_scs_to_aud())
        self.H_LM = dsp.Filter(size=(self.__Room.RIR_length, n_M, n_L), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_LM.assign_value(self.__Room.get_lds_to_mcs())
        self.H_LA = dsp.Filter(size=(self.__Room.RIR_length, n_A, n_L), nfft=self.nfft, requires_grad=False, alias_decay_db=alias_decay_db)
        self.H_LA.assign_value(self.__Room.get_lds_to_aud())

        # Virtual room
        self.G = dsp.parallelGain(size=(self.n_L,), nfft=self.nfft, alias_decay_db=alias_decay_db)
        self.G.assign_value(torch.ones(self.n_L))
        fir_matrix = dsp.Filter(size=(FIR_order, self.n_L, self.n_M), nfft=self.nfft, requires_grad=True, alias_decay_db=alias_decay_db) # map=lambda x: x/torch.norm(x, 'fro'),
        # wgn_rev = WGN_reverb(matrix_size=(self.n_L,), t60=wgn_RT, samplerate=self.fs)
        # wgn_matrix = dsp.parallelFilter(size=wgn_rev.shape, nfft=self.nfft, alias_decay_db=alias_decay_db)
        # wgn_matrix.assign_value(wgn_rev)

        self.V_ML = OrderedDict([ ('U', fir_matrix) ]) # , ('R', wgn_matrix)

        # Feedback loop
        self.F_MM = system.Shell(
            core=self.__FL_iteration(self.V_ML, self.G, self.H_LM),
            input_layer=nn.Sequential(dsp.Transform(lambda x: x.diag_embed()), dsp.FFT(self.nfft))
            )
        self.set_G_to_GBI()

    # ================================== FORWARD PATH ==================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes one iteration of the feedback loop.

            **Args**:
                x (torch.Tensor): input signal.

            **Returns**:
                torch.Tensor: Depending on the input, it can be the microphones signals or the feedback loop matrix.
            
            **Usage**:
                If x is a vector of unit impulses of size (_, n_M), the output is a vector of size (_, n_M) representing the microphones signals.
                If x is a diagonal matrix of unit impulses of size (_, n_M, n_M), the output is a matrix of size (_, n_M, n_M) representing the feedback loop matrix.
                The first dimension of vectors and matrices depends on input_layer and output_layer of the Shell instance self.F_MM.
        """
        return self.F_MM(x)
    
    # ================================== OTHER METHODS ==================================

    # ------------------------ General gain methods ------------------------

    def get_G(self) -> nn.Module:
        r"""
        Return the general gain value in linear scale.

            **Returns**:
                torch.Tensor: general gain value (linear scale).
        """
        return self.G

    def set_G(self, g: float) -> None:
        r"""
        Set the general gain value in linear scale.

            **Args**:
                g (float): new general gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a float."
        self.G.assign_value(g*torch.ones(self.n_L))

    def get_current_GBI(self) -> torch.Tensor:
        r"""
        Return the Gain Before Instability (GBI) value in linear scale.
        The GBI is always computed with respect to a system general gain G=1.

            **Returns**:
                torch.Tensor: GBI value (linear scale).
        """
        # Save current G value
        g_current = self.G.param.data[0].clone()

        # Reset G module
        self.G.assign_value(torch.ones(self.n_L))

        # Compute the gain before instability
        maximum_eigenvalue = torch.max(get_magnitude(self.get_F_MM_eigenvalues()))
        gbi = torch.reciprocal(maximum_eigenvalue)

        # Restore G value
        self.set_G(g_current)

        return gbi
    
    def set_G_to_GBI(self) -> None:
        r"""
        Set the system general gain to match the current system GBI in linear scale.
        """
        # Compute the current gain before instability
        gbi = self.get_current_GBI()

        # Apply the current gain before instability to the system general gain module
        self.set_G(gbi)

    # ------------------------------------------------------------------------------
    # ---------------------------- Virtual Room methods ----------------------------

    def normalize_U(self, value: float=1.0) -> None:
        r"""
        Normalize the dsp matrix IRs to a Frobenius norm of given value.

            **Args**:
                value (float, optional): value to normalize the matrix IRs. Defaults to 1.0.
        """
        self.V_ML['U'].assign_value(self.V_ML['U'].param.data/torch.norm(self.V_ML['U'].param.data, 'fro') * value)

    # ------------------------------------------------------------------------------
    # ------------------------ Feedback-loop matrix methods ------------------------

    def __FL_iteration(self, v_ml: OrderedDict, g: nn.Module, h_lm: nn.Module)-> nn.Sequential:
        r"""
        Generate a Series object instance representing one iteration of the feedback loop.

            **Args**:
                - h_lm (nn.Module): Feedback paths from loudspeakers to microphones.
                - v_ml (OrderedDict): Virtual room components.
                - g (nn.Module): General gain.

            **Returns**:
                nn.Sequential: Series implementing one feedback-loop iteration.
        """
        f_mm = nn.Sequential()
        for key,value in v_ml.items():
            f_mm.add_module(key, value)

        f_mm.add_module('G', g)
        f_mm.add_module('H_LM', h_lm)
        
        return system.Series(f_mm)

    def get_F_MM_eigenvalues(self) -> torch.Tensor:
        r"""
        Compute the eigenvalues of the feedback-loop matrix.

            **Returns**:
                torch.Tensor: eigenvalues.
        """
        with torch.no_grad():

            # Compute eigenvalues
            evs = get_eigenvalues(self.F_MM.get_freq_response(fs=self.fs, identity=True))

        return evs

    # ------------------------------------------------------------------------------
    # ---------------------------- Full system methods -----------------------------

    def __create_system(self) -> tuple[system.Shell, system.Shell]:
        f"""
        Create the full system's Natural and Electroacoustic paths.

            **Returns**:
                tuple[Shell, Shell]: Natural and Electroacoustic paths as Shell object instances.
        """
        # Build digital signal processor
        processor = nn.Sequential()
        for key,value in self.V_ML.items():
            processor.add_module(key, value)
        processor.add_module('G', self.G)
        # Build feedback loop
        feedback_loop = system.Recursion(fF=processor, fB=self.H_LM)
        # Build the electroacoustic path
        ea_components = nn.Sequential(OrderedDict([
            ('H_SM', self.H_SM),
            ('FeedbackLoop', feedback_loop),
            ('H_LA', self.H_LA)
        ]))
        ea_path = system.Shell(core=ea_components, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        # Build the natural path
        nat_path = system.Shell(core=self.H_SA, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        return nat_path, ea_path
    
    def system_simulation(self) -> torch.Tensor:
        r"""
        Simulate the full system. Produces the system impulse response.

            **Returns**:
                torch.Tensor: system impulse response.
        """
        with torch.no_grad():
            # Generate the paths
            nat_path, ea_path = self.__create_system()
            # Compute system response
            y = nat_path.get_time_response() + ea_path.get_time_response()

        return y


# ============================================== Plots =============================================

def plot_evs_distributions(evs_1: torch.Tensor, evs_2: torch.Tensor, fs: int, nfft: int, lower_f_lim: float, higher_f_lim: float, label1: str='Initialized', label2: str='Optimized') -> None:
    r"""
    Plot the magnitude distribution of the given eigenvalues.

        **Args**:
            evs_init (torch.Tensor): First set of eigenvalues to plot.
            evs_opt (torch.Tensor): Second set of eigenvalues to plot.
            fs (int): Sampling frequency.
            nfft (int): FFT size.
            label1 (str, optional): Label for the first set of eigenvalues. Defaults to 'Initialized'.
            label2 (str, optional): Label for the second set of eigenvalues. Defaults to 'Optimized'.
    """
    idx1 = int(nfft/fs * lower_f_lim)
    idx2 = int(nfft/fs * higher_f_lim)
    evs = mag2db(torch.cat((evs_1.unsqueeze(-1), evs_2.unsqueeze(-1)), dim=len(evs_1.shape))[idx1:idx2,:,:])
    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    plt.figure(figsize=(7,6))
    ax = plt.subplot(1,1,1)
    colors = ['tab:blue', 'tab:orange']
    for i in range(evs.shape[2]):
        evst = torch.reshape(evs[:,:,i], (evs.shape[0]*evs.shape[1], -1)).squeeze()
        evst_max = torch.max(evst, 0)[0]
        ax.boxplot(evst.numpy(), positions=[i], widths=0.7, showfliers=False, notch=True, patch_artist=True, boxprops=dict(edgecolor='k', facecolor=colors[i]), medianprops=dict(color='k'))
        ax.scatter([i], [evst_max], marker="o", s=35, edgecolors='black', facecolors=colors[i])
    plt.ylabel('Magnitude in dB')
    plt.xticks([0,1], [label1, label2])
    plt.xticks(rotation=90)
    ax.yaxis.grid(True)
    plt.title(f'Eigenvalue Magnitude Distribution\nbetween {lower_f_lim} Hz and {higher_f_lim} Hz')
    plt.tight_layout()

def plot_spectrograms(y_1: torch.Tensor, y_2: torch.Tensor, fs: int, nfft: int=2**10, label1='Initialized', label2='Optimized', title='System Impulse Response Spectrograms') -> None:
    r"""
    Plot the spectrograms of the system impulse responses at initialization and after optimization.
    
        **Args**:
            - y_1 (torch.Tensor): First signal to plot.
            - y_2 (torch.Tensor): Second signal to plot.
            - fs (int): Sampling frequency.
            - nfft (int, optional): FFT size. Defaults to 2**10.
            - label1 (str, optional): Label for the first signal. Defaults to 'Initialized'.
            - label2 (str, optional): Label for the second signal. Defaults to 'Optimized'.
            - title (str, optional): Title of the plot. Defaults to 'System Impulse Response Spectrograms'.
    """
    Spec_init,f,t = mlab.specgram(y_1.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=nfft//8)
    Spec_opt,_,_ = mlab.specgram(y_2.detach().squeeze().numpy(), NFFT=nfft, Fs=fs, noverlap=nfft//8)

    max_val = max(Spec_init.max(), Spec_opt.max())
    Spec_init = Spec_init/max_val
    Spec_opt = Spec_opt/max_val
    

    plt.rcParams.update({'font.family':'serif', 'font.size':20, 'font.weight':'heavy', 'text.usetex':True})
    fig,axes = plt.subplots(2,1, sharex=False, sharey=True, figsize=(7,5), constrained_layout=True)
    
    plt.subplot(2,1,1)
    plt.pcolormesh(t, f, 10*np.log10(Spec_init), cmap='magma', vmin=-100, vmax=0)
    plt.ylim(20, 20000)
    plt.yscale('log')
    plt.title(label1)

    plt.subplot(2,1,2)
    im = plt.pcolormesh(t, f, 10*np.log10(Spec_opt), cmap='magma', vmin=-100, vmax=0)
    plt.ylim(20, 20000)
    plt.yscale('log')
    plt.title(label2)

    fig.supxlabel('Time in seconds')
    fig.supylabel('Frequency in Hz')
    fig.suptitle(title)

    cbar = fig.colorbar(im, ax=axes[:], aspect=20)
    cbar.set_label('Magnitude in dB')
    ticks = np.arange(-100, 1, 20)
    cbar.ax.set_ylim(-100, 0)
    cbar.ax.set_yticks(ticks, ['-100','-80','-60','-40','-20','0'])

def plot_coupling(rirs):
    norm_val = torch.norm(rirs, 'fro')

    rms = torch.sqrt(torch.sum(torch.pow(rirs, 2), dim=(0))/rirs.shape[0])
    plt.figure()
    image = plt.imshow(rms)
    plt.ylabel('Microphone')
    plt.yticks(np.arange(0, rirs.shape[1]), labels=np.arange(1,rirs.shape[1]+1))
    plt.xlabel('Loudspeaker')
    plt.xticks(np.arange(0, rirs.shape[2]), labels=np.arange(1,rirs.shape[2]+1))
    plt.colorbar(mappable=image)

    # new_rirs = rirs/rms
    # new_rirs = new_rirs/torch.norm(new_rirs, 'fro') * norm_val
    # rms = torch.sqrt(torch.sum(torch.pow(new_rirs, 2), dim=(0))/rirs.shape[0])
    # plt.figure()
    # image = plt.imshow(rms)
    # plt.ylabel('Microphone')
    # plt.yticks(np.arange(0, rirs.shape[1]), labels=np.arange(1,rirs.shape[1]+1))
    # plt.xlabel('Loudspeaker')
    # plt.xticks(np.arange(0, rirs.shape[2]), labels=np.arange(1,rirs.shape[2]+1))
    # plt.colorbar(mappable=image)
    plt.show(block=True)

def plot_stuff(samplerate, nfft, rtfs, fl_rtfs, evs):

    f_axis = torch.linspace(0, samplerate//2, nfft//2+1)
    rtfs_peak = torch.max(torch.max(torch.abs(rtfs), dim=2)[0], dim=1)[0]
    rtfs_mean = torch.mean(torch.abs(rtfs), dim=(1,2))
    rtfs_ptmr = rtfs_peak / rtfs_mean
    if len(fl_rtfs.shape) < 3:
        fl_rtfs = fl_rtfs.unsqueeze(-1)
    fl_rtfs_peak = torch.max(torch.max(torch.abs(fl_rtfs), dim=2)[0], dim=1)[0]
    fl_rtfs_mean = torch.mean(torch.abs(fl_rtfs), dim=(1,2))
    fl_rtfs_ptmr = fl_rtfs_peak/fl_rtfs_mean
    evs_peak = torch.max(torch.abs(evs), dim=1)[0]
    evs_mean = torch.mean(torch.abs(evs), dim=1)
    evs_ptmr = evs_peak/evs_mean

    plt.figure()
    plt.subplot(231)
    plt.plot(f_axis, mag2db(rtfs_peak), label='Maximum')
    plt.plot(f_axis, mag2db(rtfs_mean), label='Mean')
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.title('Room transfer functions - magnitude')
    plt.subplot(234)
    plt.plot(f_axis, mag2db(rtfs_ptmr))
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.subplot(232)
    plt.plot(f_axis, mag2db(fl_rtfs_peak))
    plt.plot(f_axis, mag2db(fl_rtfs_mean))
    plt.ylim(-50,0)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.title('Feedback-loop transfer functions - magnitude')
    plt.subplot(235)
    plt.plot(f_axis, mag2db(fl_rtfs_ptmr))
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.subplot(233)
    plt.plot(f_axis, mag2db(evs_peak))
    plt.plot(f_axis, mag2db(evs_mean))
    plt.ylim(-50,0)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.title('Eigenvalues - magnitude')
    plt.subplot(236)
    plt.plot(f_axis, mag2db(evs_ptmr))
    plt.ylim(-20,30)
    plt.xlim(20,20000)
    plt.xscale('log')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)

# ==================================================================================================
# =========================================== Auxiliary ============================================

class AA_RIRs(object):
    def __init__(self, dir: str, n_S: int, n_L: int, n_M: int, n_A: int, fs: int) -> None:
        r"""
        Room impulse response wrapper class.
        These room impulse responses were measured in the listening room called Otala inside
        the Aalto Acoustics Lab in the Aalto University's Otaniemi campus, Espoo, Finland.

            **Args**:
                - dir (str): Path to the room impulse responses.
                - n_S (int): Number of sources. Defaults to 1.
                - n_L (int): Number of loudspeakers. Defaults to 1.
                - n_M (int): Number of microphones. Defaults to 1.
                - n_A (int): Number of audience members. Defaults to 1.
                - fs (int): Sample rate [Hz].
        """
        object.__init__(self)
        # assert n_S == 1, "Only one source is supported."
        # assert n_L <= 13, "Only up to 13 loudspeakers are supported."
        # assert n_M <= 4, "Only up to 4 microphones are supported."
        # assert n_A == 1, "Only one audience member is supported."
        
        self.n_S = n_S
        self.n_L = n_L
        self.n_M = n_M
        self.n_A = n_A
        self.fs = fs
        self.dir = dir
        self.__RIRs, self.RIR_length = self.__load_rirs()

    def __load_rirs(self) -> torch.Tensor:
        r"""
        Give the directory, loads the corresponding RIRs.

            **Returns**:
                torch.Tensor: RIRs. dtype=torch.float32, shape = (RIRs_length, n_M, n_L).
        """

        rirs_length = 288000 # Read this from dataset
        sr = 96000          # Read this from dataset
        new_rirs_length = int(self.fs * rirs_length/sr) # I should infer this from the resample

        src_to_aud = torch.zeros(new_rirs_length, self.n_A, self.n_S)
        for i in range(self.n_A):
            for j in range(self.n_S):
                w = torchaudio.load(f"{self.dir}/StageAudience/R{i+1:03d}_S{j+1:03d}.wav")[0]
                if self.fs != sr:
                    w = torchaudio.transforms.Resample(sr, self.fs)(w)
                src_to_aud[:,i,j] = w.permute(1,0).squeeze()

        src_to_sys = torch.zeros(new_rirs_length, self.n_M, self.n_S)
        for i in range(self.n_M):
            for j in range(self.n_S):
                w = torchaudio.load(f"{self.dir}/StageSystem/R{i+1:03d}_S{j+1:03d}.wav")[0]
                if self.fs != sr:
                    w = torchaudio.transforms.Resample(sr, self.fs)(w)
                src_to_sys[:,i,j] = w.permute(1,0).squeeze()

        sys_to_aud = torch.zeros(new_rirs_length, self.n_A, self.n_L)
        for i in range(self.n_A):
            for j in range(self.n_L):
                w = torchaudio.load(f"{self.dir}/SystemAudience/R{i+1:03d}_S{j+1:03d}.wav")[0]
                if self.fs != sr:
                    w = torchaudio.transforms.Resample(sr, self.fs)(w)
                sys_to_aud[:,i,j] = w.permute(1,0).squeeze()

        sys_to_sys = torch.zeros(new_rirs_length, self.n_M, self.n_L)
        for i in range(self.n_M):
            for j in range(self.n_L):
                w = torchaudio.load(f"{self.dir}/SystemSystem/E{j+1:03d}_R{i+1:03d}_M03.wav")[0]
                if self.fs != sr:
                    w = torchaudio.transforms.Resample(sr, self.fs)(w)
                sys_to_sys[:,i,j] = w.permute(1,0).squeeze()[0:new_rirs_length]

        rirs = OrderedDict([
            ('src_to_aud', src_to_aud),
            ('src_to_sys', src_to_sys),
            ('sys_to_aud', sys_to_aud),
            ('sys_to_sys', sys_to_sys)
        ])
        return rirs, new_rirs_length
    
    def get_scs_to_aud(self) -> torch.Tensor:
        r"""
        Returns the sources to audience RIRs

            **Returns**:
                torch.Tensor: Sources to audience RIRs. shape = (15000, n_A, n_S).
        """
        return self.__RIRs['src_to_aud']

    def get_scs_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the sources to microphones RIRs

            **Returns**:
                torch.Tensor: Sources to microphones RIRs. shape = (15000, n_M, n_S).
        """
        return self.__RIRs['src_to_sys']
    
    def get_lds_to_aud(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to audience RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to audience RIRs. shape = (15000, n_A, n_L).
        """
        return self.__RIRs['sys_to_aud']

    def get_lds_to_mcs(self) -> torch.Tensor:
        r"""
        Returns the loudspeakers to microphones RIRs

            **Returns**:
                torch.Tensor: Loudspeakers to microphones RIRs. shape = (15000, n_M, n_L).
        """
        return self.__RIRs['sys_to_sys']


# class MSE_evs(nn.Module):
#     def __init__(self, iter_num: int, freq_points: int):
#         r"""
#         Mean Squared Error (MSE) loss function for Active Acoustics.
#         To reduce computational complexity (i.e. the number of eigendecompositions computed),
#         the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
#         The subset is selected randomly ensuring that all frequency points are considered once and only once.

#             **Args**:
#                 - iter_num (int): Number of iterations per epoch.
#                 - freq_points (int): Number of frequency points.
#         """
#         super().__init__()

#         self.max_index = freq_points

#         self.iter_num = iter_num
#         self.idxs = torch.randperm(freq_points)
#         self.evs_per_iteration = torch.ceil(torch.tensor(freq_points / self.iter_num, dtype=torch.float))
#         self.interval_count = 0

#     def forward(self, y_pred, y_true):
#         r"""
#         Compute the MSE loss function.
            
#             **Args**:
#                 - y_pred (torch.Tensor): Predicted eigenvalues.
#                 - y_true (torch.Tensor): True eigenvalues.

#             **Returns**:
#                 torch.Tensor: Mean Squared Error.
#         """
#         # Get the indexes of the frequency-point subset
#         idxs = self.__get_indexes()
#         # Get the eigenvalues
#         evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
#         evs_true = y_true[:,idxs,:]
#         mse = torch.mean(torch.square(evs_pred - evs_true))
#         return mse

#     def __get_indexes(self):
#         r"""
#         Get the indexes of the frequency-point subset.

#             **Returns**:
#                 torch.Tensor: Indexes of the frequency-point subset.
#         """
#         # Compute indeces
#         idx1 = np.min([int(self.interval_count*self.evs_per_iteration), self.max_index-1])
#         idx2 = np.min([int((self.interval_count+1) * self.evs_per_iteration), self.max_index])
#         idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
#         # Update interval counter
#         self.interval_count = (self.interval_count+1) % (self.iter_num)
#         return idxs
    
class MSE_evs_mod(nn.Module):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, lowest_f: float, crossover_freq: float, highest_f: float):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__()

        assert(lowest_f >= 0)
        nyquist = samplerate//2
        assert(highest_f <= nyquist)

        min_freq_point = int(lowest_f/nyquist * freq_points)
        max_freq_point = int(highest_f/nyquist * freq_points)
        crossover_point = int(crossover_freq/nyquist * freq_points)

        ratio = (max_freq_point - min_freq_point) / (crossover_point - min_freq_point)
        self.freq_points = max_freq_point - min_freq_point
        self.max_index = self.freq_points

        self.weights = ( torch.sigmoid(torch.linspace(7, -7*ratio, self.freq_points+min_freq_point)) * 4 ) + 1

        self.iter_num = iter_num
        self.idxs = torch.randperm(self.freq_points) + min_freq_point
        self.evs_per_iteration = torch.ceil(torch.tensor(self.freq_points / self.iter_num, dtype=torch.float))
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        evs_true = y_true[:,idxs,:]
        difference = evs_pred - evs_true
        if evs_pred.shape[2] > 4:
            mask = difference > 0.0
            difference[mask] = difference[mask] * 3
            # difference[~mask] = torch.abs(difference[~mask]) # ** 2
        weights = self.weights[idxs].unsqueeze(0).unsqueeze(-1).repeat(1,1,evs_true.shape[-1])
        mse = torch.mean(torch.square(torch.abs(difference) * weights))
        return mse

    def __get_indexes(self):
        r"""
        Get the indexes of the frequency-point subset.

            **Returns**:
                torch.Tensor: Indexes of the frequency-point subset.
        """
        # Compute indeces
        idx1 = np.min([int(self.interval_count*self.evs_per_iteration), self.max_index-1])
        idx2 = np.min([int((self.interval_count+1) * self.evs_per_iteration), self.max_index])
        idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
        # Update interval counter
        self.interval_count = (self.interval_count+1) % (self.iter_num)
        return idxs
    
class minimize_evs(nn.Module):
    def __init__(self, iter_num: int, freq_points: int, samplerate: int, lowest_f: float, crossover_freq: float, highest_f: float):
        r"""
        Mean Squared Error (MSE) loss function for Active Acoustics.
        To reduce computational complexity (i.e. the number of eigendecompositions computed),
        the loss is applied only on a subset of the frequecy points at each iteration of an epoch.
        The subset is selected randomly ensuring that all frequency points are considered once and only once.

            **Args**:
                - iter_num (int): Number of iterations per epoch.
                - freq_points (int): Number of frequency points.
        """
        super().__init__()

        assert(lowest_f >= 0)
        nyquist = samplerate//2
        assert(highest_f <= nyquist)

        min_freq_point = int(lowest_f/nyquist * freq_points)
        max_freq_point = int(highest_f/nyquist * freq_points)
        crossover_point = int(crossover_freq/nyquist * freq_points)

        ratio = (max_freq_point - min_freq_point) / (crossover_point - min_freq_point)
        self.freq_points = max_freq_point - min_freq_point
        self.max_index = self.freq_points

        self.weights = ( torch.sigmoid(torch.linspace(7, -7*ratio, self.freq_points+min_freq_point)) * 4 ) + 1

        self.iter_num = iter_num
        self.idxs = torch.randperm(self.freq_points) + min_freq_point
        self.evs_per_iteration = torch.ceil(torch.tensor(self.freq_points / self.iter_num, dtype=torch.float))
        self.interval_count = 0

    def forward(self, y_pred, y_true):
        r"""
        Compute the MSE loss function.
            
            **Args**:
                - y_pred (torch.Tensor): Predicted eigenvalues.
                - y_true (torch.Tensor): True eigenvalues.

            **Returns**:
                torch.Tensor: Mean Squared Error.
        """
        # Get the indexes of the frequency-point subset
        idxs = self.__get_indexes()
        # Get the eigenvalues
        evs_pred = get_magnitude(get_eigenvalues(y_pred[:,idxs,:,:]))
        mse = torch.mean(torch.square(torch.abs(evs_pred)))
        return mse

    def __get_indexes(self):
        r"""
        Get the indexes of the frequency-point subset.

            **Returns**:
                torch.Tensor: Indexes of the frequency-point subset.
        """
        # Compute indeces
        idx1 = np.min([int(self.interval_count*self.evs_per_iteration), self.max_index-1])
        idx2 = np.min([int((self.interval_count+1) * self.evs_per_iteration), self.max_index])
        idxs = self.idxs[torch.arange(idx1, idx2, dtype=torch.int)]
        # Update interval counter
        self.interval_count = (self.interval_count+1) % (self.iter_num)
        return idxs
    
class preserve_reverb_energy_mod(nn.Module):
    def __init__(self, idxs):
        super().__init__()
        self.idxs = idxs
    def forward(self, y_pred, y_target, model):
        freq_response = model.F_MM._Shell__core.U.freq_response
        return torch.mean(torch.pow(torch.abs(freq_response[self.idxs])-torch.abs(y_target.squeeze()[self.idxs].unsqueeze(2).repeat(1,1,16)), 2))
    
def save_model_params(model: system.Shell, filename: str='parameters'):
    r"""
    Retrieves the parameters from a given model and saves them in .mat format.

        **Parameters**:
            model (Shell): The Shell class containing the FDN.
            filename (str): The name of the file to save the parameters without file extension. Defaults to 'parameters'.
        **Returns**:
            dict: A dictionary containing the FDN parameters.
                - 'FIR_matrix' (ndarray): The FIR matrix.
                - 'WGN_reverb' (ndarray): The WGN reverb.
                - 'G' (ndarray): The general gain.
                - 'H_LM' (ndarray): The loudspeakers to microphones RIRs.
                - 'H_LA' (ndarray): The loudspeakers to audience RIRs.
                - 'H_SM' (ndarray): The sources to microphones RIRs.
                - 'H_SA' (ndarray): The sources to audience RIRs.
    """

    param = {}
    param['FIR_matrix'] = model.V_ML['U'].param.squeeze().detach().clone().numpy()
    # param['WGN_reverb'] = model.V_ML['R'].param.squeeze().detach().clone().numpy()
    param['G'] = model.G.param.squeeze().detach().clone().numpy()
    param['H_LM'] = model.H_LM.param.squeeze().detach().clone().numpy()
    param['H_LA'] = model.H_LA.param.squeeze().detach().clone().numpy()
    param['H_SM'] = model.H_SM.param.squeeze().detach().clone().numpy()
    param['H_SA'] = model.H_SA.param.squeeze().detach().clone().numpy()
    
    scipy.io.savemat(os.path.join(args.train_dir, filename + '.mat'), param)

    return param


# ==================================================================================================
# ============================================ Example =============================================

def example_AA(args) -> None:
    r"""
    Active Acoustics training test function.
    Training results are plotted showing the difference in performance between the initialized model and the optimized model.
    The model parameters are saved to file.
    You can modify the number of microphones (should be set between 1 and 4) and the number of loudspeakers (should be set between 1 and 13).
    Please use n_S = 1 and  n_A = 1.
    Measured room impulse responses for additional source and/or audience positions are not available.

        **Args**:
            A dictionary or object containing the necessary arguments for the function.
    """

    # --------------------- Parameters ------------------------
    samplerate = 96000                  # Sampling frequency
    nfft = samplerate*2                 # FFT size

    stage = 0                           # Number of stage sources
    microphones = 16                    # Number of microphones
    loudspeakers = 32                   # Number of loudspeakers
    audience = 0                        # Number of audience receivers

    FIR_order = 2**8                    # FIR filter order
    rirs_dir = './rirs/LA-lab'          # Path to the room impulse responses
    equalized_system = False

    lowest_f = 20                       # Lower frequency limit for the loss function
    crossover_loss = 500               # Crossover frequency for the loss function weights (sigmoid)
    crossover_dataset = 8000            # Crossover frequency for the target (limit between flat and evs-like)
    highest_f = 16000                   # Upper frequency limit for the loss function
    

    # ------------------- Model Definition --------------------
    model = AA(
        n_S = stage,
        n_M = microphones,
        n_L = loudspeakers,
        n_A = audience,
        room_name = rirs_dir,
        fs = samplerate,
        nfft = nfft,
        FIR_order = FIR_order,
        wgn_RT = None,
        alias_decay_db=0
    )
    
    # ------------- Performance at initialization -------------
    # Normalize for fair comparison
    model.normalize_U()
    # We initialize the model to an instable state.
    gbi_init = model.get_current_GBI()
    model.set_G(db2mag(mag2db(gbi_init) + 0))
    # Performance metrics
    evs_init = model.get_F_MM_eigenvalues().squeeze(0)
    # ir_init = model.system_simulation().squeeze(0)

    rirs = model.H_LM.param.data
    rtfs = torch.fft.rfft(rirs, nfft, dim=0)
    fl_rtfs = model.F_MM.get_freq_response(identity=True).squeeze(0)
    f_axis = torch.linspace(0, samplerate//2, nfft//2+1)
    # plot_stuff(samplerate=samplerate, nfft=nfft, rtfs=rtfs, fl_rtfs=fl_rtfs, evs=evs_init)
    # plot_coupling(rirs)

    # Save the model parameters
    save_model_params(model, filename='AA_parameters_init')

    # ----------------- Initialize dataset --------------------
    dataset = DatasetColorless_mod(
        input_shape = (args.batch_size, nfft//2+1, microphones),
        target_shape = (args.batch_size, nfft//2+1, microphones),
        rtfs = evs_init,
        crossover_freq = crossover_dataset,
        highest_f = highest_f,
        nyquist = samplerate/2,
        equalized_system = equalized_system,
        expand = args.num,
        device = args.device
        )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size, split=args.split, shuffle=False)

    # ------------- Initialize training process ---------------
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        patience_delta=args.patience_delta,
        train_dir=args.train_dir,
        device=args.device
    )
    # criterion1 = minimize_evs(iter_num=args.num, freq_points=nfft//2+1, samplerate=samplerate, lowest_f=lowest_f, crossover_freq=crossover_loss, highest_f=highest_f)
    # trainer.register_criterion(criterion1, 1)
    # criterion2 = preserve_reverb_energy_mod(idxs=torch.arange(0, nfft//2+1))
    # trainer.register_criterion(criterion2, 1, requires_model=True)
    criterion = MSE_evs_mod(iter_num=args.num, freq_points=nfft//2+1, samplerate=samplerate, lowest_f=lowest_f, crossover_freq=crossover_loss, highest_f=highest_f)
    trainer.register_criterion(criterion, 1)
    
    # ------------------- Train the model --------------------
    trainer.train(train_loader, valid_loader)

    # ------------ Performance after optimization ------------
    # Normalize for fair comparison
    model.normalize_U()
    # Performance metrics
    evs_opt = model.get_F_MM_eigenvalues().squeeze(0)
    # ir_opt = model.system_simulation().squeeze(0)

    # Save the model parameters
    save_model_params(model, filename='AA_parameters_optim')
    
    # ------------------------ Plots -------------------------
    plot_evs_distributions(get_magnitude(evs_init), get_magnitude(evs_opt), samplerate, nfft, lowest_f, highest_f)
    # plot_spectrograms(ir_init[:,0], ir_opt[:,0], samplerate)

    f_axis = torch.linspace(0, samplerate//2, nfft//2+1)
    plt.figure()
    plt.subplot(211)
    plt.plot(f_axis, mag2db(torch.max(torch.abs(evs_init), dim=1)[0]))
    plt.plot(f_axis, mag2db(torch.max(torch.abs(evs_opt), dim=1)[0]))
    plt.subplot(212)
    plt.plot(f_axis, mag2db(torch.mean(torch.abs(evs_init), dim=1)))
    plt.plot(f_axis, mag2db(torch.mean(torch.abs(evs_opt), dim=1)))

    virtual_room = system.Shell(model.V_ML)
    filters = virtual_room.get_time_response(identity=True).squeeze(0)[0:FIR_order,:,:]
    ftfs = virtual_room.get_freq_response(identity=True).squeeze(0)
    t_axis = torch.linspace(0, filters.shape[0]/samplerate, filters.shape[0])
    f_axis = torch.linspace(0, samplerate//2, nfft//2+1)
    plt.figure()
    plt.subplot(211)
    plt.plot(t_axis, filters[:,4,1])
    plt.plot(t_axis, filters[:,12,3])
    plt.subplot(212)
    plt.plot(f_axis, mag2db(torch.abs(ftfs[:,4,1])))
    plt.plot(f_axis, mag2db(torch.abs(ftfs[:,12,3])))
    plt.show(block=True)

    to_save = torch.zeros((FIR_order*microphones, loudspeakers))
    for i in range(loudspeakers):
        for j in range(microphones):
            to_save[j*FIR_order:(j+1)*FIR_order,i] = filters[:,i,j]
    torchaudio.save('./test_in_reaper/fir_LA_noEQ_16-16_parallel.wav', to_save, channels_first=False, sample_rate=samplerate)

    return None

###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--num', type=int, default=2**5,help = 'dataset size')
    parser.add_argument('--device', type=str, default='cpu', help='device to use for computation') # try msp
    parser.add_argument('--split', type=float, default=0.9, help='split ratio for training and validation')
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, help='directory to save training results')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--patience_delta', type=float, default=0.01, help='Minimum improvement in validation loss to be considered as an improvement')
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    #----------------- Parse the arguments ----------------
    args = parser.parse_args()

    # make output directory
    if args.train_dir is not None:
        if not os.path.isdir(args.train_dir):
            os.makedirs(args.train_dir)
    else:
        args.train_dir = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.train_dir)

    # save arguments
    with open(os.path.join(args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # Run examples
    example_AA(args)