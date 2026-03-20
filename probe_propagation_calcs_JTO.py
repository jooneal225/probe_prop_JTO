import numpy as np
from scipy.optimize import curve_fit


def calculate_res_and_dof(energy, det_distance_m, det_pixel_um, img_size):
    lambda_nm = 1.2398 / energy
    pixel_size = lambda_nm * 1.e-9 * det_distance_m / (img_size * det_pixel_um * 1e-6)
    depth_of_field = lambda_nm * 1.e-9 / (img_size / 2 * det_pixel_um * 1.e-6 / det_distance_m) ** 2

    return pixel_size, depth_of_field

def guassian(data, height, center, width, background):

    return background + height*np.exp(-(data-center)**2/(2*width**2))


def gaussian_fit(data):
    X = np.arange(data.size)
    xc = np.sum(X * data) / np.sum(data)
    width = np.abs(np.sqrt(np.abs(np.sum((X - xc) ** 2 * data) / np.sum(data))))
    try:
        popt, pcov = curve_fit(guassian,X,data,p0 = [data.max(),xc,width,data[0:5].mean()])
        y_fit = guassian(X,popt[0],popt[1],popt[2],popt[3])
    except:
        popt, pcov = X, X
        y_fit = data/data

    return popt, pcov, y_fit

def propagate(probe_np_array,energy,dist,dx, dy):

    """"dist,dx,dy in microns"""

    wavelength_m = 12.398*1.e-4/energy

    k = 2. * np.pi / wavelength_m
    nx, ny = np.shape(probe_np_array)
    spectrum = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(probe_np_array)))

    dkx = 2. * np.pi / (nx * dx)
    dky = 2. * np.pi / (ny * dy)

    skx = dkx * nx / 2
    sky = dky * ny / 2

    kproj_x = np.linspace(-skx, skx - dkx, nx)
    kproj_y = np.linspace(-sky, sky - dky, ny)
    kx, ky = np.meshgrid(kproj_y, kproj_x)

    phase = np.sqrt(k ** 2 - kx ** 2 - ky ** 2) * dist

    spectrum *= np.exp(1j * phase)
    array_prop = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(spectrum)))

    return array_prop


def propagate_probe(probe_array,energy,nx_size_m,ny_size_m, start_um=-50,end_um=50,step_size_um=1):


    # load the probe file
    prb_ini = probe_array
    nx, ny = np.shape(prb_ini)

    # propagation distance and steps
    num_steps = int((end_um - start_um) / step_size_um) + 1
    projection_points = np.linspace(start_um,
                                    end_um,
                                    num_steps)

    #print(projection_points)

    #for sigma
    sigma = np.zeros((3, num_steps))
    deviation = np.zeros((3, num_steps))

    #for fitted data
    xfits = np.zeros((num_steps, 2, nx))
    yfits = np.zeros((num_steps, 2, ny))

    #initial probe
    prb = propagate(prb_ini,
                    energy,0,
                    nx_size_m*10**6,
                    ny_size_m*10**6)

    # create an image stack with probe at each propogated distance
    prop_data = np.zeros((nx, ny, num_steps)).astype(complex)


    for i, distance in enumerate(projection_points):

        #print(i)
        tmp = propagate(prb,
                        energy,
                        distance,
                        nx_size_m*10**6,
                        ny_size_m*10**6)

        prop_data[:, :, i] = tmp

        if i == 0:
            sig_x, sig_y,data_x,data_y = probe_img_to_linefit(tmp,
                                                              gaussian_sig_init = 0.8)

        else:
            sig_x, sig_y,data_x,data_y  = probe_img_to_linefit(tmp,
                                                               gaussian_sig_init=sigma[1,i-1])

        sigma[0,i] = distance
        sigma[1, i] = sig_x
        sigma[2, i] = sig_y

        deviation[0, i] = distance

        pha = np.angle(tmp)
        if np.max(pha)-np.min(pha) > 5:
                pha[pha<0] += 2*np.pi
        deviation[1, i] = np.sqrt(np.mean((pha-np.mean(pha))**2))

        amp = np.abs(tmp)
        deviation[2, i] = np.sqrt(np.mean((amp-np.mean(amp))**2))
        xfits[i] = data_x

        yfits[i] = data_y

    return prop_data, sigma, deviation, xfits, yfits

def propagate_gpu(probe_np_array, energy, dist, dx, dy, torch):
    """GPU version of propagate() using PyTorch.  dist, dx, dy in microns.

    Args:
        torch: the torch module, passed in by the caller to avoid a
               module-level import (allows graceful fallback to CPU path).
    Returns:
        numpy complex128 array — identical interface to propagate().
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wavelength_m = 12.398e-4 / energy
    k = 2. * np.pi / wavelength_m
    nx, ny = probe_np_array.shape

    prb_t = torch.as_tensor(probe_np_array, dtype=torch.complex128, device=device)
    spectrum = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(prb_t)))

    dkx = 2. * np.pi / (nx * dx)
    dky = 2. * np.pi / (ny * dy)
    skx = dkx * nx / 2
    sky = dky * ny / 2

    kproj_x = np.linspace(-skx, skx - dkx, nx)
    kproj_y = np.linspace(-sky, sky - dky, ny)
    kx_np, ky_np = np.meshgrid(kproj_y, kproj_x)  # mirrors original meshgrid order
    kx_t = torch.as_tensor(kx_np, dtype=torch.float64, device=device)
    ky_t = torch.as_tensor(ky_np, dtype=torch.float64, device=device)

    phase = torch.sqrt(k ** 2 - kx_t ** 2 - ky_t ** 2) * dist
    spectrum = spectrum * torch.exp(1j * phase)
    array_prop = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(spectrum)))

    return array_prop.cpu().numpy()


def propagate_probe_gpu(probe_array, energy, nx_size_m, ny_size_m,
                        start_um=-50, end_um=50, step_size_um=1, torch=None):
    """GPU-accelerated version of propagate_probe() using PyTorch.

    Key optimisation: the FFT of the probe is computed only once on the GPU,
    then all propagation distances are phase-multiplied and inverse-FFT'd in a
    single batched operation — rather than one FFT per distance step.

    Args:
        torch: the torch module (passed in to avoid a module-level import).
    Returns:
        (prop_data, sigma, deviation, xfits, yfits) — identical to propagate_probe().
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prb_ini = probe_array
    nx, ny = np.shape(prb_ini)
    dx = nx_size_m * 1e6   # convert m → µm (matching propagate() convention)
    dy = ny_size_m * 1e6

    num_steps = int((end_um - start_um) / step_size_um) + 1
    projection_points = np.linspace(start_um, end_um, num_steps)

    wavelength_m = 12.398e-4 / energy
    k = 2. * np.pi / wavelength_m

    # --- k-space grid (computed on CPU, then sent to GPU) ---
    dkx = 2. * np.pi / (nx * dx)
    dky = 2. * np.pi / (ny * dy)
    skx = dkx * nx / 2
    sky = dky * ny / 2
    kproj_x = np.linspace(-skx, skx - dkx, nx)
    kproj_y = np.linspace(-sky, sky - dky, ny)
    kx_np, ky_np = np.meshgrid(kproj_y, kproj_x)  # mirrors original meshgrid order
    kx_t = torch.as_tensor(kx_np, dtype=torch.float64, device=device)
    ky_t = torch.as_tensor(ky_np, dtype=torch.float64, device=device)

    # --- Step 1: dist=0 propagation — mirrors: prb = propagate(prb_ini, energy, 0, dx, dy) ---
    prb_ini_t = torch.as_tensor(prb_ini, dtype=torch.complex128, device=device)
    spectrum_ini = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(prb_ini_t)))
    # dist=0 → phase=0 → exp(0)=1, spectrum is unchanged
    prb_t = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(spectrum_ini)))

    # --- Step 2: compute spectrum of prb ONCE (reused for every distance) ---
    spectrum_prb = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(prb_t)))

    # --- Step 3: batch-compute phases for all distances simultaneously ---
    # sqrt_k2 shape: (nx, ny);  distances_t shape: (num_steps,)
    sqrt_k2 = torch.sqrt(k ** 2 - kx_t ** 2 - ky_t ** 2)
    distances_t = torch.as_tensor(projection_points, dtype=torch.float64, device=device)
    # phases shape: (num_steps, nx, ny)
    phases = sqrt_k2.unsqueeze(0) * distances_t.reshape(-1, 1, 1)

    # --- Step 4: apply phases and inverse-FFT all distances in one shot ---
    # all_spectra: (num_steps, nx, ny)
    all_spectra = spectrum_prb.unsqueeze(0) * torch.exp(1j * phases)
    # fftshift / fftn / fftshift on spatial dims only (leave num_steps dim untouched)
    all_prop = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.fftshift(all_spectra, dim=(-2, -1)),
            dim=(-2, -1),
        ),
        dim=(-2, -1),
    )  # shape: (num_steps, nx, ny)

    prop_data_np = all_prop.cpu().numpy()          # (num_steps, nx, ny)
    prop_data = prop_data_np.transpose(1, 2, 0)    # (nx, ny, num_steps) — matches original

    # --- Step 5: Gaussian fits on CPU (unchanged from original) ---
    sigma = np.zeros((3, num_steps))
    deviation = np.zeros((3, num_steps))
    xfits = np.zeros((num_steps, 2, nx))
    yfits = np.zeros((num_steps, 2, ny))

    for i, distance in enumerate(projection_points):
        tmp = prop_data_np[i]  # (nx, ny)

        if i == 0:
            sig_x, sig_y, data_x, data_y = probe_img_to_linefit(tmp, gaussian_sig_init=0.8)
        else:
            sig_x, sig_y, data_x, data_y = probe_img_to_linefit(tmp, gaussian_sig_init=sigma[1, i - 1])

        sigma[0, i] = distance
        sigma[1, i] = sig_x
        sigma[2, i] = sig_y

        deviation[0, i] = distance
        pha = np.angle(tmp)
        if np.max(pha) - np.min(pha) > 5:
            pha[pha < 0] += 2 * np.pi
        deviation[1, i] = np.sqrt(np.mean((pha - np.mean(pha)) ** 2))

        amp = np.abs(tmp)
        deviation[2, i] = np.sqrt(np.mean((amp - np.mean(amp)) ** 2))

        xfits[i] = data_x
        yfits[i] = data_y

    return prop_data, sigma, deviation, xfits, yfits


def probe_img_to_linefit(prb_image, gaussian_sig_init = 0.8):

    nx, ny = np.shape(prb_image)
    #axis for projection 1D
    proj_x = np.arange(nx, dtype = np.float64)
    proj_y = np.arange(ny,dtype = np.float64)

    x_fit_data = np.zeros((2,nx))
    y_fit_data = np.zeros((2, ny))

    # find the max points in the image and get the line profile at that point
    ix, iy = np.where(np.abs(prb_image) == np.nanmax(np.abs(prb_image)))
    prb_intensity = (np.abs(prb_image)) ** 2
    line_tmp_x = np.squeeze(prb_intensity.sum(1))
    line_tmp_y = np.squeeze(prb_intensity.sum(0))

    x_fit_data[0] = line_tmp_x
    y_fit_data[0] = line_tmp_y

    popt, pcov, y_fit = gaussian_fit(line_tmp_y/line_tmp_y.max())
    sigma_y = np.abs(popt[2])
    popt, pcov, x_fit = gaussian_fit(line_tmp_x/line_tmp_x.max())
    sigma_x = np.abs(popt[2])

    x_fit_data[1] = x_fit
    y_fit_data[1] = y_fit

    return sigma_x, sigma_y, x_fit_data, y_fit_data