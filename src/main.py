import torch
import torch.optim as optim
import matplotlib.pyplot as plt
cuda = torch.cuda.is_available()
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR

from models import *
from utilities import *
from train_loss import *
from test_loss import *

def s9_run_main(num_epochs, maxlr):

    # Data Augmentation & data loader stuff to be handled
    batch_size = 512
    trainloader, testloader = S9_CIFAR10_data_prep(batch_size)

    # Creating tensorboard writer
    img_save_path = '/content/gdrive/MyDrive/EVA8_P1_S9/'
    tb_writer = create_tensorboard_writer(img_save_path)

    # Creating plot object
    plot = cifar10_plots(img_save_path, tb_writer)

    # Displaying train data
    plot.plot_cifar10_train_imgs(trainloader)

    # Displaying torch summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = basic_attn_model().to(device)
    summary(model, input_size=(3, 32, 32))

    # Adding model graph to tensor-board
    img = torch.ones(1, 3, 32, 32)
    img = img.to(device)
    tb_writer.add_graph(model, img)

    # Note : Plot not coming out correctly when ran via module. Hence will find maxlr outside the main & pass it as arg

    # Finding Max LR using range test
    #model = basic_attn_model().to("cuda")
    #lrfinder = LRRangeFinder(model=model, epochs=3, start_lr=1e-2, end_lr=1e-1, tb_writer= tb_writer,
    #                         dataloader=trainloader, device=device, img_save_path=img_save_path)
    #max_lr = lrfinder.findLR()
    max_lr = maxlr
    print(f'max_lr is {max_lr}')

    # Training the model for fixed epochs
    EPOCHS = num_epochs
    model = basic_attn_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(trainloader), epochs=EPOCHS, 
                           pct_start = 5/24, div_factor=10, final_div_factor=1)
    stats = ctr()
    train = train_losses(model, device, trainloader, stats, optimizer, EPOCHS)
    test  = test_losses(model, device, testloader, stats, EPOCHS)
    print(f'Initial LR : {scheduler.get_lr()}')
    print(f'Total steps: {scheduler.total_steps}')

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch+1}')
        train.s9_train(epoch, scheduler, tb_writer, L1_factor=0.0005)
        test.s9_test(epoch, tb_writer)

    details = counters

    # Plot OneCycle-LR Curve
    plot_onecyclelr_curve(details, img_save_path)

    # Displaying 20 misclassified images
    num_images = 25
    plot.plot_cifar10_misclassified(details, num_images)

    # Plotting train & test accuracies and losses
    plt.figure(figsize=(12, 8))
    plt.title(f"Train Losses")
    plt.plot(details['train_loss'])
    plt.savefig(f'{img_save_path}train_loss.jpg')

    plt.figure(figsize=(12,8))
    plt.title(f"Train Accuracy")
    plt.plot(details['train_acc'])
    plt.savefig(f'{img_save_path}train_acc.jpg')

    plt.figure(figsize=(12,8))
    plt.title(f"Test Losses")
    plt.plot(details['test_loss'])
    plt.savefig(f'{img_save_path}test_loss.jpg')

    plt.figure(figsize=(12,8))
    plt.title(f"Test Accuracy")
    plt.plot(details['test_acc'])
    plt.savefig(f'{img_save_path}test_acc.jpg')

    return f' s9_run_main() ended successfully '

def s10_run_main(num_epochs):

    # Data Augmentation & data loader stuff to be handled
    batch_size = 128
    trainloader, testloader = S10_CIFAR10_data_prep(batch_size)

    # Creating tensorboard writer
    img_save_path = '/content/gdrive/MyDrive/EVA8_P1_S10/'
    tb_writer = create_tensorboard_writer(img_save_path)

    # Creating plot object
    plot = cifar10_plots(img_save_path, tb_writer)

    # Displaying train data
    plot.plot_cifar10_train_imgs(trainloader)

    # Displaying torch summary
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = ViT(image_size=32,patch_size= 4,num_classes=10, dim=49,depth=6,heads=8,mlp_dim=147,numb_patch=8, 
                dropout=0.1,emb_dropout=0.1)
    summary(model, input_size=(3, 32, 32))

    # Adding model graph to tensor-board
    img = torch.ones(1, 3, 32, 32)
    img = img.to(device)
    tb_writer.add_graph(model, img)

    # Training the model for fixed epochs
    EPOCHS = num_epochs
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    stats = ctr()
    train = train_losses(model, device, trainloader, stats, optimizer, EPOCHS)
    test  = test_losses(model, device, testloader, stats, EPOCHS)

    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch+1}')
        train.s10_train(epoch, tb_writer)
        test.s10_test(epoch, tb_writer)

    details = counters

    # Displaying 20 misclassified images
    num_images = 25
    plot.plot_cifar10_misclassified(details, num_images)

    # Plotting train & test accuracies and losses
    plt.figure(figsize=(12, 8))
    plt.title(f"Train Losses")
    plt.plot(details['train_loss'])
    plt.savefig(f'{img_save_path}train_loss.jpg')

    plt.figure(figsize=(12,8))
    plt.title(f"Train Accuracy")
    plt.plot(details['train_acc'])
    plt.savefig(f'{img_save_path}train_acc.jpg')

    plt.figure(figsize=(12,8))
    plt.title(f"Test Losses")
    plt.plot(details['test_loss'])
    plt.savefig(f'{img_save_path}test_loss.jpg')

    plt.figure(figsize=(12,8))
    plt.title(f"Test Accuracy")
    plt.plot(details['test_acc'])
    plt.savefig(f'{img_save_path}test_acc.jpg')

    return f' s10_run_main() ended successfully '