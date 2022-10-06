import math
import numpy as np
import tifffile
import zarr
import cv2
import os
import matplotlib.pyplot as plt
import torch

'''
        These functions converts a 2D or 3D image array (dimenion 3 for color 
        channels) into an array of image patches. The linearize_patches() function
        accepts the image as the input, along with the dimensions of the patches, 
        if padding is to be used, and if there is any overlap in the image patches. By
        default, padding is used, and there is no overlap. 
        
        Because the patches are now arranged in a linear array, they can easily be used
        for multiprocessing before the image is reconstructed. This format is also
        similar to that used in Pytorch, and may be easier to use for model.predict().
        
        The workflow for creating patches that are 300 x 300 is as follows:
    
# Creates the array of patches
patches_img, patches_info, N_patches = linearize_patches(target, 300, pad = True, overlap = 0)

# Pre-populate the array of modified image patches
desired_dtype = 'uint8'
patches_mod = np.zeros_like(patches_img, dtype = desired_dtype)

# Perform function on each patch
for k in range(N_patches):
    patches_mod[k,:,:,:] =   function(patches_img[k, :, :, :]).astype(desired_dtype)

# Reconstruct the image    
reconstruct = reconstruct_from_linear_patches(patches_mod, patches_info)


        Alternatively, the functions can be used to create a dictionary of image patches. 
        This method has the advantage in that the array of modified patches does
        not need to be pre-populated, and the dtype does not need to be declared. 
        This may be easier to use for parallel processing.


patch_dict, Npatches  = linearize_patches(target, 50, overlap = 15, dictout = True)

for k in range(Npatches):
    patch_dict['patches'][k] = (patch_dict['patches'][k].astype(float)*k/Npatches).astype('uint8')

reconstruct = reconstruct_from_linear_patches(patch_dict)
plt.imshow(reconstruct)


'''



def linearize_patches(img, dims, pad = True, overlap = 0, outputtype = "numpy", only_tissue = False):
    '''
    Inputs an image of dimenion (H,W,C) and the desired patch dimenions (dims), 
    and outputs an array of dimension (N, dims, dims, C).
    img    - Input image
    dims   - Length and width of the output patches
    pad    - True pads the img with black space so that the the entire image
             is present in the patches, if needed.
           - False cuts off the bottom and right edges.
           
    outputtype - numpy, dict, memmap_loc           
           
    '''   
    
    
    if outputtype == "memmap_loc":
        
        assert os.path.isfile(img) 
        
        store = tifffile.imread(img, key = 0, aszarr=True)
        z  = zarr.open(store, mode='r')
        if len(z.shape) == 2:
            C = 1
            H, W = img.shape
            original_shape = (H, W, C)
        else:
            H, W, C = original_shape = z.shape
        dtype = z.dtype
        store.close()        
    else:
        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)
        H, W, C = original_shape = img.shape
        dtype = img.dtype

            
    h1, h2, w1, w2 = [[0], [], [0], []]
    
    while True:
        h2.append(h1[-1]+dims)
        
        if h2[-1] > H: break
        else: h1.append(h2[-1] - overlap*2)
    
    while True:
        w2.append(w1[-1]+dims)
        
        if w2[-1] > W: break
        else: w1.append(w2[-1] - overlap*2)
    
    if pad == False :
        if h2[-1] > H: 
            h1 = h1[:-1]
            h2 = h2[:-1]
        if w2[-1] > W: 
            w1 = w1[:-1]
            w2 = w2[:-1]
            
            
    N_patches_H = len(h1)
    N_patches_W = len(w1)  
    
    k=0
    h1_p, h2_p, w1_p, w2_p = patch_locations = [[], [], [], []]
    for i in range(N_patches_H):
        for j in range(N_patches_W):
            h1_p.append(h1[i])
            h2_p.append(h2[i])
            w1_p.append(w1[j])
            w2_p.append(w2[j])
            k+=1
            
    if only_tissue == True and outputtype == 'memmap_loc':

        store = tifffile.imread(img, key = 3, aszarr=True)
        z  = zarr.open(store, mode='r')
        real_img = z[:,:,:]
        mask = tissue_mask(real_img, thresh = 20)
        store.close()
        real_img[:, :, 0][mask == 0] = real_img[:, :, 0][mask == 0]*0.2
        real_img[:, :, 1][mask == 0] = real_img[:, :, 1][mask == 0]*0.2
        real_img[:, :, 2][mask == 0] = real_img[:, :, 2][mask == 0]*0.2
        
        plt.imshow(real_img)
        plt.show()
        
        
        ratio = mask.shape[0]/H
        
        k_list_tissue = []
        for k in range(len(h1_p)):
            patch_mask = mask[ int(h1_p[k]*ratio): int(h2_p[k]*ratio), int(w1_p[k]*ratio): int(w2_p[k]*ratio)]
            if np.sum(patch_mask)/patch_mask.size > 0.01:
                # print(np.sum(mask[ int(h1_p[k]*ratio): int(h2_p[k]*ratio), int(w1_p[k]*ratio): int(w2_p[k]*ratio)]))
                k_list_tissue.append(k)
            
            
    if not outputtype == "memmap_loc":
        if pad == True:
            img_ = np.zeros([h2[-1], w2[-1], C], dtype = dtype)
            img_[:H, :W, :] = img 
        else:
            img_ = img        
        
        patches_img  = np.empty([N_patches_H*N_patches_W,dims, dims, C], dtype = img.dtype)  
        k = 0
        for i in range(N_patches_H):
            for j in range(N_patches_W):
                patches_img[k, :,:, :] = img_[h1[i]:h2[i] , w1[j]:w2[j] , :]
                k +=1


    patches_info = {}
    patches_info['patch_locations'] = patch_locations
    patches_info['original_shape' ] = original_shape
    patches_info['pad'            ] = pad
    patches_info['overlap'        ] = overlap
    patches_info['dims'           ] = dims
    patches_info['Npatches'       ] = N_patches_H*N_patches_W
    patches_info['outputtype'     ] = outputtype
    patches_info['only_tissue'    ] = only_tissue
    
    
    if outputtype == "numpy":
        return patches_img, patches_info, patches_info['Npatches']
    
    elif outputtype == "dict":
        
        patches_info['patches'] = {}
        for k in range(patches_img.shape[0]):
            patches_info['patches'][k] = patches_img[k, :,:, :]
        
        return patches_info, patches_info['Npatches']
    
    elif outputtype == "memmap_loc":
        patches_info['patches'      ] = {}
        patches_info['filepath'     ] = img
        patches_info['dtype'        ] = dtype
        patches_info['k_list_tissue'] = k_list_tissue
        
        return patches_info, patches_info['Npatches']
            
        

def pull_patch_memmap(patches_info, k):

    H,W,_                  = patches_info['original_shape' ]
    h1_p, h2_p, w1_p, w2_p = patches_info['patch_locations']
    pad                    = patches_info['pad'            ]
    overlap                = patches_info['overlap'        ]
    dims                   = patches_info['dims'           ] 
    (H, W, C)              = patches_info['original_shape' ]
    dtype                  = patches_info['dtype'          ]
    filepath               = patches_info['filepath'       ]
    
    patch_out = np.zeros([dims, dims, C], dtype = dtype)
    
    store = tifffile.imread(filepath, key = 0, aszarr=True)
    z  = zarr.open(store, mode='r')
    
    h1, h2, w1, w2 = h1_p[k], np.min([h2_p[k], H]), w1_p[k], np.min([w2_p[k], W])
    if C > 1:
        patch_out[:h2-h1, :w2-w1, :] = z[h1:h2, w1:w2,:]
    elif C == 1:
        patch_out[:h2-h1, w2-w1, 1] = z[h1:h2, w1:w2]
    
    store.close()
    
    return patch_out
    
    
    

def reconstruct_from_linear_patches(patches_mod, patches_info = None):
    '''
    Inputs the array of modified patches and outputs a reconstructed
    image.
    patches_info - Dictionary of parameters used to reconstruct the image.
    Note - The numpy of color channels output is the number of color channels
    in the patches_mod array.
    '''   
    
    if type(patches_mod) == dict:
        patches_info = patches_mod
        del patches_mod
        
        
        first_idx = list(patches_info['patches'].keys())[0]
               
        patches_mod_ = np.zeros( [patches_info['Npatches']] + list(patches_info['patches'][first_idx].shape) , dtype= patches_info['patches'][first_idx].dtype)
     
        
        
        if patches_info['outputtype'     ] == 'dict':
            for k in range(patches_info['Npatches']):
                patches_mod_[k, :, :, :] = patches_info['patches'][k]
                
        elif patches_info['outputtype'     ] == 'memmap_loc' and patches_info['only_tissue'    ] == True:
            ct = 0
            for k in patches_info['k_list_tissue']:
                patches_mod_[ct, :, :, :] = patches_info['patches'][k]
                ct+=1
        
        patches_mod = patches_mod_
        del patches_mod_
        
        
    
    H,W,_                  = patches_info['original_shape' ]
    h1_p, h2_p, w1_p, w2_p = patches_info['patch_locations']
    pad                    = patches_info['pad'            ]
    overlap                = patches_info['overlap'        ]
    dims                   = patches_info['dims'           ] 
    
    if len(patches_mod.shape) == 3:
        patches_mod = np.expand_dims(patches_mod, 3)
    N_patches, _, _, C = patches_mod.shape
    
    
    reconstruct = np.zeros( (np.max(h2_p), np.max(w2_p), C), dtype=patches_mod.dtype)
    
    if  patches_info['outputtype'] == 'memmap_loc':
        k_list = patches_info['k_list_tissue']
    else:
        k_list = range(N_patches)
    
    
    for k in k_list:
        h1, h2, w1, w2 = h1_p[k]+overlap, h2_p[k]-overlap, w1_p[k]+overlap, w2_p[k]-overlap
        y1, y2, x1, x2 = overlap, dims-overlap, overlap, dims-overlap
        
        if h1_p[k] == 0: 
            h1 = h1_p[k]
            y1 = 0
        if w1_p[k] == 0: 
            w1 = w1_p[k]
            x1 = 0
        # if pad == False:
        if h2_p[k] == np.max(h2_p): 
            h2 = h2_p[k]
            y2 = dims
        if w2_p[k] == np.max(w2_p): 
            w2 = w2_p[k]
            x2 = dims
        
        
        reconstruct[h1:h2, w1:w2, :]  =  patches_mod[k_list.index(k), y1:y2, x1:x2,:]               

        
        # if h1_p[k] == 0 and w1_p[k] == 0:
        #     reconstruct[h1:h2-overlap, w1:w2-overlap, :]  =  patches_mod[k, :-overlap,:-overlap,:]               
        # if h1_p[k] == 0 :
        #     reconstruct[h1_p[k]:h2_p[k]-overlap, w1_p[k]+overlap:w2_p[k]-overlap, :]  =  patches_mod[k, :-overlap,overlap:-overlap,:]               
        # if w1_p[k] == 0:
        #     reconstruct[h1_p[k]+overlap:h2_p[k]-overlap, w1_p[k]:w2_p[k]-overlap, :]  =  patches_mod[k, overlap:-overlap,:-overlap,:]               
        # else:
        #     reconstruct[h1_p[k]+overlap:h2_p[k]-overlap, w1_p[k]+overlap:w2_p[k]-overlap, :]  =  patches_mod[k, overlap:-overlap,overlap:-overlap,:]               
    
    if pad == True:
        if reconstruct.shape[0] > H:
            reconstruct = reconstruct[:H, :, :]
        if reconstruct.shape[1] > W:
            reconstruct = reconstruct[:, :W, :]
 
    elif pad == False:
        reconstruct = reconstruct[:np.max(h2_p), :np.max(w2_p), :]
    
    if C == 1:
        np.squeeze(reconstruct, 2)    
    
    
    return reconstruct
    






def tissue_mask(img, thresh=10, rounds = 3, iterations = 1):

    img1 = 255-cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img1[img1==255] = 0
    
    p_cutoff = np.percentile(   img1[img1>0]  , 2) + thresh
    
    
    img_bw = (img1>p_cutoff).astype('uint8')
    
    kernel = np.ones((4, 4), np.uint8) 
    
    for i in range(rounds):
        img_dilation = cv2.dilate(img_bw.copy(), kernel, iterations=iterations) 
        img_erosion = cv2.erode(img_dilation, kernel, iterations=iterations)
    
    return img_erosion





def torch_deconv(img, device = None, W_est = 'No_West', I_0 = 250, allow_negatives= False):
    
    if device == None:
        if torch.cuda.is_available():
            device = 'cuda'
        else: device = 'cpu'
    assert device  == 'cuda' or device == 'cpu'
    
    if type(W_est) == str:
        # Assign W_est to Hematoxylin and Eosin matrix
        W_est = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    if np.linalg.norm(W_est[:, 2]) <= 1e-16:
        stain0 = W_est[:, 0]
        stain1 = W_est[:, 1]
        stain2 = np.cross(stain0, stain1)
        wc = np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T   
    else: wc = W_est
    

    wc = wc / np.max(wc)   
    

    Q = np.linalg.inv(wc).astype('float32')
    

    
    img_t = torch.from_numpy(img).to(device)
    
    
    # m = convert_image_to_matrix(im_rgb)[:3]
    is_matrix = img_t.ndim == 2
    if is_matrix == 2:
        m = img_t.T[:3] 
    else: 
        m = img_t.reshape((-1, img_t.shape[-1])).T[:3]


    # rgb_to_sda
    # if m.ndim == 2:
    #     m = m.T
    m = m + 1.0
    if I_0 is None:  # rgb_to_od compatibility
        
        I_0 = 256
    if not allow_negatives:
        m = torch.minimum(m, torch.ones(m.shape, device = device)*I_0)
    im_sda = -torch.log((m+1)/(1.*I_0)) * 255/np.log(I_0)
    

    # sda_deconv = np.dot(Q, sda_fwd)
    sda_deconv = torch.mm(torch.from_numpy(Q).to(device), im_sda)

    # sda_inv = sda_to_rgb(sda_deconv, 255 if I_0 is not None else None) 
    if sda_deconv.ndim == 2:
        sda_deconv = sda_deconv.T

    im_rgb = I_0 ** (1 - sda_deconv / 255.)
    sda_inv =  (im_rgb.T if sda_deconv.ndim == 2 else im_rgb) 
    
   
# reshape output
    if len(img.shape) == 2:
        StainsFloat = sda_inv
    else: 
        StainsFloat = sda_inv.T.reshape(img.shape[:-1] + (sda_inv.shape[0],))
    


    Stains = StainsFloat.clip(0, 255).to(torch.uint8).to('cpu').numpy()

    return Stains
    
def reconv(Stains, W_est = 'No_West', s = 0):
    

    if type(W_est) == str:
        # Assign W_est to Hematoxylin and Eosin matrix
        W_est = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
    
    
    Stains = 255 - Stains
    
    out =np.zeros_like(Stains)
    out[:,:,0] = 255-Stains[:,:,s]*W_est[0,s]
    out[:,:,1] = 255-Stains[:,:,s]*W_est[1,s]
    out[:,:,2] = 255-Stains[:,:,s]*W_est[2,s]
    
    return out
    


def np_deconv(img, device = None, W_est = 'No_West', I_0 = 250, allow_negatives= False):
    
    
    if type(W_est) == str:
        # Assign W_est to Hematoxylin and Eosin matrix
        W_est = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])

    if np.linalg.norm(W_est[:, 2]) <= 1e-16:
        stain0 = W_est[:, 0]
        stain1 = W_est[:, 1]
        stain2 = np.cross(stain0, stain1)
        wc = np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T   
    else: wc = W_est
    

    wc = wc / np.max(wc)   
    

    Q = np.linalg.inv(wc).astype('float32')
    

    is_matrix = img.ndim == 2
    if is_matrix == 2:
        m = img.T[:3] 
    else: 
        m = img.reshape((-1, img.shape[-1])).T[:3]


    m = m.astype(float) + 1
    if I_0 is None:  # rgb_to_od compatibility
        
        I_0 = 256
    if not allow_negatives:
        m = np.minimum(m, I_0)
    im_sda = -np.log((m+1)/(1.*I_0)) * 255/np.log(I_0)
    
    sda_deconv = np.dot(Q, im_sda)

    if sda_deconv.ndim == 2:
        sda_deconv = sda_deconv.T

    im_rgb = I_0 ** (1 - sda_deconv / 255.)
    sda_inv =  (im_rgb.T if sda_deconv.ndim == 2 else im_rgb) 
    
    if len(img.shape) == 2:
        StainsFloat = sda_inv
    else: 
        StainsFloat = sda_inv.T.reshape(img.shape[:-1] + (sda_inv.shape[0],))
    
    Stains = StainsFloat.clip(0, 255).astype('uint8')

    return Stains