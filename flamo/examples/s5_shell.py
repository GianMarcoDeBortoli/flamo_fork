import argparse
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from flamo.processor import dsp, system
from flamo.functional import mag2db, get_magnitude, get_eigenvalues, signal_gallery
from flamo.optimize.dataset import DatasetColorless, load_dataset
from flamo.optimize.trainer import Trainer


def s4_e0():
    """
    In all previous examples, we always defined input and output layers for the DSP.
    This is because the DSP needs to interface with the dataset input and to provide the output in
    a format that is compatible with the loss function.
    In this section we introduce the Shell class.
    The Shell class possess three attributes: input_layer, core and output_layer.
    The core is the DSP or the model that we want to train. 
    The input_layer and the output_layer are the same ones we so far defined.
    The important property of the Shell class is that is keeps the three components separate instead
    of merging them into a single torch.nn.Sequential object or Series class instance.
    This means we can easily keep the dsp fixed and change the input and output layers according to
    the dataset and the loss function that we want to use.
    The Shell function provides dedicated methods to change the input and output layers.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 96000

    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 1
    # Filters
    filter1 = dsp.Gain(
        size=(out_ch, in_ch),
        nfft=nfft,
    )
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
    )
    filter3 = dsp.parallelFilter(
        size=(50, out_ch,),
        nfft=nfft
    )
    filter4 = dsp.parallelSVF(
        size=(out_ch,),
        n_sections=2,
        nfft=nfft,
        fs=samplerate,
    )
    filters = OrderedDict({
        'Gain': filter1,
        'Delay': filter2,
        'FIR': filter3,
        'SVF': filter4
    })

    # Shell
    my_dsp = system.Shell(core=filters)

    # ---------- DSP time and frequency responses -----------

    # Input unit impulse
    unit_imp = signal_gallery(signal_type='impulse', batch_size=1, n_samples=samplerate, n=in_ch, fs=samplerate)

    # Time response
    my_dsp.set_inputLayer(dsp.FFT(nfft=nfft))
    my_dsp.set_outputLayer(dsp.iFFT(nfft=nfft))
    imp_resp = my_dsp(unit_imp)

    # Magnitude response
    my_dsp.set_outputLayer(
        nn.Sequential(
            dsp.Transform(get_magnitude),
            dsp.Transform(mag2db)
            )
    )
    mag_resp = my_dsp(unit_imp)

    # ------------------------ Plot -------------------------
    plt.figure()
    plt.subplot(211)
    plt.plot(imp_resp.squeeze().numpy())
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.subplot(212)
    plt.plot(mag_resp.squeeze().numpy())
    plt.xlabel('Frequency bins')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

    return None

def s4_e1():
    """
    In addition to that, the Shell checks that all three components
    are comparitible in terms of attributes and input/output channels.
    Let's insert a filter in the input layer and set the wrong number of channels on purpose.
    The Shell class will tell us what the error is and where the error is located.
    It will through an error at the first erroneuos test found.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 96000

    # ------------------- DSP Definition --------------------
    in_ch = 3
    out_ch = 3
    # Filters
    filter1 = dsp.Gain(
        size=(out_ch, in_ch),
        nfft=nfft,
    )
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
    )
    filter3 = dsp.parallelFilter(
        size=(50, out_ch,),
        nfft=nfft
    )
    filter4 = dsp.parallelSVF(
        size=(out_ch,),
        n_sections=2,
        nfft=nfft,
        fs=samplerate,
    )
    filters = OrderedDict({
        'Gain': filter1,
        'Delay': filter2,
        'FIR': filter3,
        'SVF': filter4
    })

    # Layers
    in_layer1 = dsp.FFT(nfft=nfft)
    in_layer2 = dsp.Gain(size=(10, in_ch), nfft=nfft, requires_grad=False)          # NOTE: Error here
    in_layer = nn.Sequential(in_layer1, in_layer2)

    out_layer = dsp.iFFT(nfft=nfft)          # NOTE: Error here

    # Shell
    my_dsp = system.Shell(input_layer=in_layer, core=filters, output_layer=out_layer)

    return None


def s4_e2():
    """
    Finally, the Shell class provides useful functionalities to retrieve the time and frequency responses
    directly, without having to modify input and output layers manually.
    We can repeat the first example of the section using the get_time_response() and get_freq_response() methods.
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 96000

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 1
    # Filters
    filter1 = dsp.Gain(
        size=(out_ch, in_ch),
        nfft=nfft,
        requires_grad=False
    )
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=3000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    filter3 = dsp.parallelSVF(
        size=(out_ch,),
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    filters = OrderedDict({
        'Gain': filter1,
        'Delay': filter2,
        'SVF': filter3
    })

    # Shell
    my_dsp = system.Shell( core=filters )

    # ----------- DSP time a frequency responses ------------

    # Time response
    imp_resp = my_dsp.get_time_response(fs=samplerate)

    # Magnitude response
    freq_resp = my_dsp.get_freq_response(fs=samplerate)
    mag_resp = mag2db(get_magnitude(freq_resp))

    # ------------------------ Plot -------------------------
    plt.figure()
    plt.subplot(211)
    plt.plot(imp_resp.squeeze().cpu().numpy())
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.subplot(212)
    plt.plot(mag_resp.squeeze().cpu().numpy())
    plt.xlabel('Frequency bins')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

    return None

def s4_e3():
    """
    Thanks to the Shell class it is very easy to check what is happening inside the core.
    We can retrieve the time and frequency responses of:
        - the full dsp considering the final mixing (previous example)
        - the single modules considering the final mixing NOTE: not implemented yet
        - the full dsp without the final mixing (see Shell.get_time_response() documentation for more details)
        - the single modules without the final mixing NOTE: not implemented yet
    """

    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 96000

    # ------------------- DSP Definition --------------------
    in_ch = 2
    out_ch = 1
    # Filters
    filter1 = dsp.Gain(
        size=(out_ch, in_ch),
        nfft=nfft,
        requires_grad=False
    )
    filter2 = dsp.parallelDelay(
        size=(out_ch,),
        max_len=5000,
        isint=True,
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    filter3 = dsp.parallelSVF(
        size=(out_ch,),
        nfft=nfft,
        fs=samplerate,
        requires_grad=False
    )
    filters = OrderedDict({
        'Gain': filter1,
        'Delay': filter2,
        'SVF': filter3
    })

    # Shell
    my_dsp = system.Shell( core=filters )

    # ----------- DSP time a frequency responses ------------

    # Time response
    imp_resp = my_dsp.get_time_response(fs=samplerate, interior=True)

    # Magnitude response
    freq_resp = my_dsp.get_freq_response(fs=samplerate, interior=True)
    mag_resp = mag2db(get_magnitude(freq_resp))

    # ------------------------ Plot -------------------------
    plt.figure()
    for i in range(in_ch):
        plt.subplot(2, in_ch, i+1)
        plt.plot(imp_resp.squeeze().cpu().numpy()[:,i])
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title(f'Input channel {i+1}')
        plt.subplot(2, in_ch, i+2)
        plt.plot(mag_resp.squeeze().cpu().numpy()[:,i])
        plt.xlabel('Frequency bins')
        plt.ylabel('Magnitude')
        plt.grid()
    plt.tight_layout()
    plt.show()

    return None


def s4_e4(args):
    """
    In this example, we will see how flexible the Shell class can be.
    We will define the DSP just one time. We will istantiate the Shell class only once.
    We will then change input and output layers of the Shell accordingly to the type of
    training we want to perform.
    """
    # -------------- Time-frequency parameters --------------
    samplerate = 48000
    nfft = 96000

    # ------------------ Model Definition -------------------
    FIR_order = 1000
    in_ch = 2
    out_ch = 2
    my_dsp = dsp.Filter(
        size=(FIR_order, out_ch, in_ch),
        nfft=nfft,
        requires_grad=True
    )

    # Shell instance
    model = system.Shell( core=my_dsp )

    # Get the initial response for the comparison
    fr_init = model.get_freq_response(fs=samplerate, interior=False)
    interior_fr_init = model.get_freq_response(fs=samplerate, interior=True)
    evs_init = torch.linalg.eigvals(get_magnitude(model.get_freq_response(fs=samplerate, interior=True)))

    # ========================================================================================================
    # Case 1: Train the DSP to have all flat magnitude responses
    # NOTE: We need the Shell.input_layer to modify the input in order to obtain the system's frequency
    #       responses without the final mixing (see Shell.get_time_response() documentation for more details).
    #       We will use the torchTensor.diag_embed() method to create a diagonal matrix from the input tensor.
    #       We will also need the Shell.output_layer to compute the magnitude of the output frequency responses.

    # Initialize dataset
    dataset = DatasetColorless(
        in_shape=(args.batch_size, args.nfft//2+1, in_ch),
        target_shape=(args.batch_size, args.nfft//2+1, out_ch, in_ch),
        ds_len=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Initialize loss function
    criterion = torch.nn.MSELoss()

    # Interface DSP with dataset and loss function
    model.set_inputLayer(nn.Sequential(dsp.Transform(lambda x: x.diag_embed()), dsp.FFT(args.nfft)))
    model.set_outputLayer(dsp.Transform(get_magnitude))

    # Initialize training process
    trainer = Trainer(
        net=model,
        max_epochs=args.max_epochs,
        lr=args.lr,
        train_dir=args.train_dir,
        device=args.device
    )
    trainer.register_criterion(criterion, 1)
    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get the optimized response
    interior_fr_optim = model.get_freq_response(fs=samplerate, interior=True)

    plt.figure()
    for i in range(out_ch):
        for j in range(in_ch):
            plt.subplot(out_ch, in_ch, i * in_ch + j + 1)
            plt.plot(mag2db(get_magnitude(interior_fr_init[0, :, i, j]).detach()).numpy(), label='Init')
            plt.plot(mag2db(get_magnitude(interior_fr_optim[0, :, i, j]).detach()).numpy(), label='Optim')
            plt.xlabel('Frequency bins')
            plt.ylabel('Magnitude')
            plt.grid()
    plt.legend()
    plt.suptitle("Filter's interior magnitude responses")
    plt.tight_layout()
    plt.show()

    # ========================================================================================================
    # Case 2: Train the DSP to have all eigenvalues equal to 1 in magnitude
    # NOTE: We will keep the same Shell.input_layer as before.
    #       We will also need the Shell.output_layer to compute first the eigenvalues of the complex output
    #       frequency responses, and then the magnitude of the eigenvalues.

    # Change the dataset
    dataset = DatasetColorless(
        in_shape=(args.batch_size, args.nfft//2+1, in_ch),
        target_shape=(args.batch_size, args.nfft//2+1, out_ch), # The target has the a different shape now
        ds_len=args.num,
        device=args.device,
    )
    train_loader, valid_loader = load_dataset(dataset, batch_size=args.batch_size)

    # Interface DSP with new loss function
    model.set_outputLayer(nn.Sequential(dsp.Transform(get_eigenvalues), dsp.Transform(get_magnitude)))

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get the optimized response
    evs_optim = torch.linalg.eigvals(get_magnitude(model.get_freq_response(fs=samplerate, interior=True)))

    plt.figure()
    for i in range(out_ch):
            plt.subplot(out_ch, 1, i+1)
            plt.plot(mag2db(torch.abs(evs_init[0, :, i])).detach().numpy(), label='Init')
            plt.plot(mag2db(torch.abs(evs_optim[0, :, i])).detach().numpy(), label='Optim')
            plt.xlabel('Frequency bins')
            plt.ylabel('Magnitude')
            plt.grid()
    plt.legend()
    plt.suptitle("Filters eigenvalues")
    plt.tight_layout()
    plt.show()

    # ========================================================================================================
    # Case 3: Train the DSP to have flat output responses
    # NOTE: Finally, we want to include the final mixing in the optimization.
    #       We will modify the Shell.input_layer accordingly.
    #       We will also remove the eigenvalue computation from the Shell.output_layer.
    #       In this case we can also keep the last DatasetColorless instance.

    # Interface DSP with dataset and loss function
    model.set_inputLayer(dsp.FFT(args.nfft))
    model.set_outputLayer(dsp.Transform(get_magnitude))

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Get the optimized response
    fr_optim = model.get_freq_response(fs=samplerate, interior=False)

    plt.figure()
    for i in range(out_ch):
            plt.subplot(out_ch, 1, i+1)
            plt.plot(mag2db(get_magnitude(fr_init[0, :, i]).detach()).numpy(), label='Init')
            plt.plot(mag2db(get_magnitude(fr_optim[0, :, i]).detach()).numpy(), label='Optim')
            plt.xlabel('Frequency bins')
            plt.ylabel('Magnitude')
            plt.grid()
    plt.legend()
    plt.suptitle("Filters magnitude responses")
    plt.tight_layout()
    plt.show()

    return None

###########################################################################################

if __name__ == '__main__':

    # Define system parameters and pipeline hyperparameters
    parser = argparse.ArgumentParser()
    
    #----------------------- Dataset ----------------------
    parser.add_argument('--num', type=int, default=2**10,
                        help = 'dataset size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--split', type=float, default=0.8,
                        help='training / validation split')
    parser.add_argument('--shuffle', action='store_false',
                        help='if true, shuffle the data in the dataset at every epoch')
    
    #---------------------- Training ----------------------
    parser.add_argument('--train_dir', type=str, default=None,
                        help ='path to output directory')
    parser.add_argument('--device', default='cpu',
                        help='training device')
    parser.add_argument('--max_epochs', type=int, default=10, 
                        help='maximum number of training epochs')
    parser.add_argument('--log_epochs', action='store_true',
                        help='Store met parameters at every epoch')
    
    #---------------------- Optimizer ---------------------
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    
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
    # s4_e0()
    s4_e1()
    # s4_e2()
    # s4_e3()
    # s4_e4(args)