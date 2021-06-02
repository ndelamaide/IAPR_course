import numpy as np
import imutils
import skimage.io
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, opening, disk
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


#--------------- PRE-PROCESSING ------------------------------

def crop(image, top=100, bottom=4400):
    ''' Crops image from top to bottom
    '''
    return image[top:bottom, :, :]


def decompose_channels(image):
    ''' Decompose image into greyscale and RGB channels
    '''
    
    gray = skimage.color.rgb2gray(image)
    red = image[:, :, 0]
    green = image[:, :,1]
    blue = image[:, :,2]
    
    return [gray, red, green, blue]


def norm_binary(image):
    ''' Normalizes the image of the round  and returns the binary version
    '''
    
    # Contrast stretching
    a, b = np.percentile(image, (0, 50))
    norm_im = rescale_intensity(image, in_range=(a,b))
    
    # Apply otsu method for threshold
    #threshold = threshold_otsu(norm_im)
    #binary = norm_im > threshold
    
    # Apply multiotsu
    threshold = threshold_multiotsu(norm_im)
    digitized = np.digitize(norm_im, bins=threshold)
    
    return digitized

def binarize_cards(cards, channel=2):
    ''' Rescales intensity and binarizes the image of the cards on specified channel
    Returns : Binarized card images
    '''
    
    bin_cards = cards
    
    for i in range(len(cards)):
        
        if cards[i].size!=0:
        
            #Normalizes card
            norm = rescale_intensity(cards[i][:, :, channel])
        
            #Digitizes card
            th = threshold_multiotsu(norm)
            dig = np.digitize(norm, bins=th)
            bin_cards[i] = dig == 0 
        
    return bin_cards

def card_morpho(card):
    ''' Performs closing on image of card
    Returns : Modified image 
    '''
    
    im_card = closing(card, disk(3))
    
    return im_card

def suite_morpho(suite):
    ''' Performs closing and opening on image of suite
    Returns : Modified image 
    '''
    
    im_suite = closing(suite, square(2))
    im_suite = opening(im_suite, disk(2))
    
    return im_suite


def rank_morpho(rank):
    ''' Performs closing and opening on image of the rank
    Returns : Modified image 
    '''
    
    im_rank = closing(rank, disk(2))
    im_rank = opening(im_rank, disk(4))
    
    return im_rank

#--------------- CARD EXTRACTION ------------------------------

def find_bbox(image, threshold=25000, plot=True):
    ''' Finds the bboxes of the objects (here the cards and dealer token)
    Returns the number of bboxes 
    and a dictionary of tuples (list_bbox_bounds, bbox_area, bbox_center)
    '''
    
    closed = closing(image, square(2))
    
    cleared = clear_border(closed)

    # label image regions
    label_image = label(cleared, background=0, connectivity=1)
    
    # Store bboxes
    bboxes = {}
    num = 0

    for region in regionprops(label_image):
        
        # take regions with large enough area
        if  region.area >= threshold:
            
            # Compute area of bbox
            minr, minc, maxr, maxc = region.bbox

            c = maxc - minc
            r = maxr - minr
            area = c * r
            
            # Compute bbox center
            xc = (minr + maxr) // 2
            yc = (minc + maxc) // 2
            
            bboxes[num] = ([minr, minc, maxr, maxc], area, [xc, yc])
            num += 1
    
    if plot:
        # Plot overlay of image regions and bboxes
        plot_overlays(image, label_image, threshold)
    
    return num, bboxes


def get_centers(bboxes):
    ''' Get the centers of the bboxes of the cards only
    '''
    
    x_centers = []
    y_centers = []
    
    for i in range(len(bboxes.keys())):
        if bboxes[i][1] >= 200000: # Check if it's a card
            
            xc, yc = bboxes[i][2]
            
            x_centers.append(xc)
            y_centers.append(yc)
            
    return np.array(x_centers), np.array(y_centers)


def compute_distance(x1, y1, x2, y2):
    ''' Computes euclidian distance between (x1, y1) and (x2, y2)
    '''
    return np.sqrt(np.power(x1 - x2, 2)+np.power(y1 - y2, 2))


def remove_dup_bboxes(num, bboxes, threshold):
    ''' Remove bboxes inside others or overlaping
    Remove a bbox if center is closer than threshold of another center
    Only works if two overlapping bboxes
    '''
    
    # Keeps track of compared bboxes
    compared = np.zeros((num, num))
    
    # Index of bboxes to remove
    idx_remove = []
    
    for i in range(num):
        if bboxes[i][1] >= 200000: # Check if it's a card
            
            # Get first bbox center
            x1, y1 = bboxes[i][2]
            
            for j in range(num):
                
                # Compare only if never compared before
                if ((compared[i][j]==0) & (compared[j][i]==0) 
                     & (i!=j) & (bboxes[j][1] >= 200000)):
                    
                    # Get second bbox center
                    x2, y2 = bboxes[j][2]
                    
                    # Check if centers are too close
                    if compute_distance(x1, y1, x2, y2) <= threshold:
                        
                        # Record one with smaller area to remove later
                        area_i = bboxes[i][1]
                        area_j = bboxes[j][1]
                        
                        if area_i > area_j:
                            idx_remove.append(j)
                        else:
                            idx_remove.append(i)
                    
                    # Record that we compared i and j
                    compared[i][j] = 1
                    compared[j][i] = 1
    
    # Keep non overlaping bboxes
    bbox_filtered = {} 
    idx = 0
    for i in range(num):
        if i not in idx_remove:
            bbox_filtered[idx] = bboxes[i]
            idx += 1
    
    return idx, bbox_filtered

    
def rearrange_boxes(bboxes):
    ''' Rearrange bboxes in the correct order
    Returns : Dictionary with bbox for each player, None if bbox not found
    '''
    
    # 0: Player1, 1: Player2 etc... 4: Dealer token
    boxes = {0: None, 1: None, 2: None, 3: None, 4: None}
    
    # If we have 1 dealer token and 4 cards
    if len(bboxes.keys()) == 5:
        xc, yc = get_centers(bboxes)
    
        min_x = np.min(xc)
        max_x = np.max(xc)
        min_y = np.min(yc)
        max_y = np.max(yc)
    
        for i in range(len(bboxes.keys())):
        
            # Check for dealer
            if bboxes[i][1] < 200000:
                boxes[4] = bboxes[i]
                
            else:
                
                x1, y1, x2, y2 = bboxes[i][0]
                x_c, y_c = bboxes[i][2]
                
                if x_c == min_x:
                    boxes[2] = bboxes[i]
                if x_c == max_x:
                    boxes[0] = bboxes[i]
                if y_c == min_y:
                    boxes[3] = bboxes[i]
                if y_c == max_y:
                    boxes[1] = bboxes[i]
                    
    # We have found less than 4 cards                
    else:
        
        # Dimensions of image
        height = 4300
        width = 3456
        
        # Divide height in 3 zones
        height_lim1 = height // 3
        height_lim2 = 2 * height_lim1
        
        # Divide width in 3 zones
        width_lim1 = width // 3
        width_lim2 = 2 * width_lim1
        
        # Go through each bbox and find it's position
        for i in range(len(bboxes.keys())):
            
            # Check for dealer
            if bboxes[i][1] < 200000:
                boxes[4] = bboxes[i]
                
            else:
                
                # Get centers
                x_c, y_c = bboxes[i][2]
                
                # If it's in top part, player 3
                if x_c < height_lim1:
                    boxes[2] = bboxes[i]
                    
                elif x_c > height_lim2: # If it's in bottom part, player1
                    boxes[0] = bboxes[i]
                    
                else: # It's either player 2 or player 4
                    
                    if y_c < width_lim1: # Player 4
                        boxes[3] = bboxes[i]
                        
                    else: # Player 2
                        boxes[1] = bboxes[i]
    return boxes


def extract_cards(img, bboxes):
    ''' From the bboxes and the image of the game, extract the image of the cards
    Returns : Dictionary of images of each card. Empty array if image not found.
    '''
    
    # 0: Player1, 1: Player2 etc...
    # We have to use empty arrays instead of None because images are arrays
    cards = {0: np.array([]), 1: np.array([]), 2: np.array([]), 3: np.array([])}
    
    for i in range(4):
        if bboxes[i] != None:
            x01, y01, x11, y11 = bboxes[i][0]
            cards[i] = img[x01:x11, y01:y11, :]
            
    return cards


def find_distances(bboxes):
    ''' Find the distance between the cards and the dealer token from the bboxes centers
    Returns : Distances of each card from the dealer card
    '''
    distances = np.empty(4)
    
    x_center_dealer, y_center_dealer = bboxes[4][2]
    
    # Find distances of cards from dealer token
    for i in range(4):
        
        if bboxes[i] != None:
            x_center_p, y_center_p = bboxes[i][2]
            dist = compute_distance(x_center_p, y_center_p, x_center_dealer, y_center_dealer)
            distances[i] = dist
        else:
            distances[i] = np.float('inf')
        
    return distances


def find_dealer(distances):
    ''' Finds dealer, i.e card with smallest distance to token
    Returns : index (0-3) corresponding to dealer player
    '''
    return np.argmin(distances)


#--------------- RANK + SUITE EXTRACTION ------------------------------


def extraction(cards):
    ''' From the image of each card extracts the rank and the suite
    Returns : The extracted suites and ranks of the cards as dictionaries
    '''
    
     # 0: Player1, 1: Player2 etc...
    # Each image has two images of its suit. We store both as list [suit1, suit2]
    extracted_suites = {0: None, 1: None, 2: None, 3: None}
    size_suite = (300, 300)
    
    # We have to use empty arrays instead of None because we directly store the images of the digits
    extracted_ranks = {0: np.array([]), 1: np.array([]), 2: np.array([]), 3: np.array([])}
    size_rank = (300, 300) # Eventually needs to be same size as MNIST images
    
    for i in range(4):
        if cards[i].size!=0:
            
            # Remove holes inside object
            card = card_morpho(cards[i])
            
            # Find bbox of objects on card
            num, boxes = find_bbox(card, 2000, True)
            
            # Get suits + digit bboxes
            rank, suites = get_suit_rank(boxes)
            
            im_suites = []
            
            # Second suit is upside down need to flip it
            rotations = [0, 180, 0, 0 , 0] # Added exta 0 in case len suites > 2
            
            # Get image of suits
            for j in range(len(suites)):
    
                dx1, dy1, dx2, dy2 = suites[j][0]
                
                # Mathematical morphology
                im_suite = suite_morpho(card[dx1-3:dx2+3, dy1-3:dy2+3])
                im_suite = resize(im_suite, size_suite)
                
                im_suites.append(imutils.rotate_bound(np.float32(im_suite), rotations[j]))
            
            # Add suites of image i to extracted_suites
            extracted_suites[i] = im_suites
            
            # Get image of rank
            dx1, dy1, dx2, dy2 = rank[0]
            
            # Mathematical morphology
            im_rank = rank_morpho(card[dx1-20:dx2+20, dy1-20:dy2+20])
            im_rank = resize(im_rank, size_suite)
            
            extracted_ranks[i] = im_rank
            
    return extracted_ranks, extracted_suites


def get_suit_rank(bboxes):
    ''' From the bboxes detected on the cards, label them as suite or rank
    Returns : the bbox of the rank and a list of the bboxes of the two suites on a card images
    '''

    # Labels bboxes as suit or rank
    
    num_bboxes = len(bboxes.keys())
    
    # Store bbox areas
    areas = np.empty(num_bboxes)
    
    for i in range(num_bboxes):
        # Sometimes card is detected, we don't keep it
        if bboxes[i][1] > 80000:
            areas[i] = -1
        else:
            areas[i] = bboxes[i][1]
        
    # Biggest area is that of digit bbox
    idx_rank = np.argmax(areas)
    
    rank = bboxes[idx_rank]
    
    suites = [bboxes[i] for i in bboxes.keys() if ((i!=idx_rank) & (areas[i]>0))]
    
    len_suites = len(suites)
    
    # If we have found 2 suites or less return then
    if len_suites <= 2:
        return rank, suites
    
    # Otherwise need to filter to keep 2. Happens when we detect a king (2 bboxes for the king symbol)
    # Merge detected suite closer to the rank as part of the rank (it's small rectangle below king)
    else:
        distances = np.empty(len_suites)
        
        for i in range(len_suites):
            distances[i] = compute_distance(rank[2][0], rank[2][1], suites[i][2][0], suites[i][2][1])
        
        # Get index of smallest distance
        idx_min = np.argmin(distances)
        
        # Merge bbox with the rank if close enough
        # We make assumption that bbox to merge is below the rank (i.e small rectangle of king)
        if distances[idx_min] < 100:
            rx1, ry1, rx2, ry2 = rank[0]
            cx1, cy1, cx2, cy2 = suites[idx_min][0]
            rank = ([rx1-18, ry1, cx2, ry2], bboxes[idx_rank][1], bboxes[idx_rank][2])
            suites_ = [suites[i] for i in range(len_suites) if (i!=idx_min)]
            
        # If it wasn't close enough, we picked up a "suite" that is not a suite
        # We make the assumption that this "suite" is an artifact that belongs to the borders of the cards
        # Thus the true suites are two closest from rank
        else:
            
            suites_ = [suites[idx_min]]
            distances[idx_min] = np.float('inf')
            suites_.append(suites[np.argmin(distances)])
            
        return rank, suites_


#--------------- PLOT FUNCTIONS ------------------------------

def plot_channels(images):
    ''' Plot the 4 channels of an image, greyscale and RGB
    '''
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 12))

    axes[0].imshow(images[0], cmap='gray')
    axes[0].set_title('Grayscale')
    axes[1].imshow(images[1], cmap='gray')
    axes[1].set_title('Red channel')
    axes[2].imshow(images[2], cmap='gray')
    axes[2].set_title('Green channel')
    axes[3].imshow(images[3], cmap='gray')
    axes[3].set_title('Blue channel')

    for i in range(4):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def plot_overlays(image, label_image, threshold=25000):
    ''' Plot overlay of image regions and bboxes
    '''
    
    # plot label and image
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    for region in regionprops(label_image):
        # take regions with large enough areas
        
        if region.area >= threshold: # 25000
            
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_bbox(image, bboxes):
    ''' Plot the bboxes and class labels over the image
    '''
    
    # Display image
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image)
    ax.axis('off')
    
    # Labels
    labels = {0: 'Player 1', 1: 'Player 2', 2: 'Player 3', 3: 'Player 4',
              4: 'Dealer'}

    for i in range(len(bboxes.keys())):
        
        if bboxes[i] != None:
            
            # Get bbox bounds
            minr, minc, maxr, maxc = bboxes[i][0]
        
            # Draw rectangle
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        
            
            ax.text(minc, minr - 50, labels[i])
        
    plt.show()


def plot_cards(cards, cmap='viridis'):
    ''' Plot the images of the cards
    '''

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    
    for i in range(4):
        if cards[i].size!=0:
            ax[i].imshow(cards[i], cmap=cmap)
        ax[i].axis('off')

    plt.show()


def plot_suites(suites, cmap='gray'):
    ''' Plot the images of the suites
    '''

    fig, ax = plt.subplots(2, 4, figsize=(10, 5))
    
    for i in range(4):
        if suites[i]!=None:
            for j in range(len(suites[i])):
                ax[j][i].imshow(suites[i][j], cmap=cmap)
        ax[0][i].axis('off')
        ax[1][i].axis('off')

    plt.show()