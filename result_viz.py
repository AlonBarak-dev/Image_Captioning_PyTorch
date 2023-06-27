import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

def show_image(inp, title=None, filename:str=None):
    """Imshow for Tensors."""
    print(inp.shape)
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    if filename is not None:
        plt.savefig("test/" + filename)
    plt.pause(5)  # pause a bit so that plots are updated


def print_examples(model, device, dataset):
    """
        Auxiliary function for validation during training.
    """
    
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("test/dog.jpg").convert("RGB")).unsqueeze(0)
    print("Validation")
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.inference(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(Image.open("test/boat.png").convert("RGB")).unsqueeze(0)
    print("Example 2 CORRECT: A small boat in the ocean")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.inference(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("test/horse.png").convert("RGB")).unsqueeze(0)
    print("Example 3 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.inference(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(Image.open("test/child.jpg").convert("RGB")).unsqueeze(0)
    print("Example 4 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.inference(test_img4.to(device), dataset.vocab))
    )
    
    model.train()


def plot_examples(model, device, dataset, model_id):
    """
        Auxiliary function for results visualization.
        Will plot & save 5 pictures, GT captions and 
        the Model prediction.
    """
    
    transform = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    model.eval()
    
    # Example 1
    img_t1 = transform(Image.open("test/dog.jpg").convert("RGB"))
    caption_1 = "Dog on a beach by the ocean"
    prediction_1 = model.inference(img_t1.unsqueeze(0).to(device), dataset.vocab)
    show_image(img_t1, f"Cap: {caption_1} \n Pred: {prediction_1}", f"dog_pred_{model_id}.jpg")
    # Example 2
    img_t2 = transform(Image.open("test/boat.png").convert("RGB"))
    caption_2 = "A small boat in the ocean"
    prediction_2 = model.inference(img_t2.unsqueeze(0).to(device), dataset.vocab)
    show_image(img_t2, f"Cap: {caption_2} \n Pred: {prediction_2}", f"boat_pred_{model_id}.jpg")
    # Example 3
    img_t3 = transform(Image.open("test/horse.png").convert("RGB"))
    caption_3 = "A cowboy riding a horse in the desert"
    prediction_3 =model.inference(img_t3.unsqueeze(0).to(device), dataset.vocab)
    show_image(img_t3, f"Cap: {caption_3} \n Pred: {prediction_3}", f"horse_pred_{model_id}.jpg")
    # Example 4
    img_t4 = transform(Image.open("test/child.jpg").convert("RGB"))
    caption_4 = "Child holding red frisbee outdoors"
    prediction_4 = model.inference(img_t4.unsqueeze(0).to(device), dataset.vocab)
    show_image(img_t4, f"Cap: {caption_4} \n Pred: {prediction_4}", f"child_pred_{model_id}.jpg")
    
    img_t5 = transform(Image.open("data/Images/44856031_0d82c2c7d1.jpg").convert("RGB"))
    caption_5 = "A dog is being squirted with water in the face outdoors"
    prediction_5 = model.inference(img_t5.unsqueeze(0).to(device), dataset.vocab)
    show_image(img_t5, f"Cap: {caption_5} \n Pred: {prediction_5}", f"dog2_pred_{model_id}.jpg")
    
    model.train()
    

    