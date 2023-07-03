#from post_processing import *
import numpy as np
from PIL import Image
import glob as gl
import numpy as np
from PIL import Image
import torch
from sklearn.preprocessing import MinMaxScaler


def accuracy_check(mask, prediction):
    ims = [mask, prediction]
    np_ims = []
    for item in ims:
        if 'str' in str(type(item)):
            item = np.array(Image.open(item))
        elif 'PIL' in str(type(item)):
            item = np.array(item)
        elif 'torch' in str(type(item)):
            item = item.detach().numpy()
        np_ims.append(item)

    compare = np.equal(np_ims[0], np_ims[1])
    accuracy = np.sum(compare)

    return accuracy/len(np_ims[0].flatten())


def accuracy_check_for_batch(masks, predictions, batch_size):
    total_acc = 0
    # print(masks.shape)
    # print(predictions.shape)
    minmax_scaler = MinMaxScaler()
    for index in range(batch_size):
        
        result = minmax_scaler.fit_transform(predictions[index][0].cpu().detach().numpy())
        result[result < 0.5] = 0
        result[result >= 0.5] = 255

        mask_ = minmax_scaler.fit_transform(masks[index][0].cpu().detach().numpy())
        mask_[mask_ < 0.5] = 0
        mask_[mask_ >= 0.5] = 255

        total_acc += accuracy_check(mask_, result)
    return total_acc/batch_size


"""
def accuracy_compare(prediction_folder, true_mask_folder):
    ''' Output average accuracy of all prediction results and their corresponding true masks.
    Args
        prediction_folder : folder of the prediction results
        true_mask_folder : folder of the corresponding true masks
    Returns
        a tuple of (original_accuracy, posprocess_accuracy)
    '''

    # Bring in the images
    all_prediction = gl.glob(prediction_folder)
    all_mask = gl.glob(true_mask_folder)

    # Initiation
    num_files = len(all_prediction)
    count = 0
    postprocess_acc = 0
    original_acc = 0

    while count != num_files:

        # Prepare the arrays to be further processed.
        prediction_processed = postprocess(all_prediction[count])
        prediction_image = Image.open(all_prediction[count])
        mask = Image.open(all_mask[count])

        # converting the PIL variables into numpy array
        prediction_np = np.asarray(prediction_image)
        mask_np = np.asarray(mask)

        # Calculate the accuracy of original and postprocessed image
        postprocess_acc += accuracy_check(mask_np, prediction_processed)
        original_acc += accuracy_check(mask_np, prediction_np)
        # check individual accuracy
        print(str(count) + 'th post acc:', accuracy_check(mask_np, prediction_processed))
        print(str(count) + 'th original acc:', accuracy_check(mask_np, prediction_np))

        # Move onto the next prediction/mask image
        count += 1

    # Average of all the accuracies
    postprocess_acc = postprocess_acc / num_files
    original_acc = original_acc / num_files

    return (original_acc, postprocess_acc)
"""

# Experimenting
if __name__ == '__main__':
    '''
    predictions = 'result/*.png'
    masks = '../data/val/masks/*.png'

    result = accuracy_compare(predictions, masks)
    print('Original Result :', result[0])
    print('Postprocess result :', result[1])
    '''
