import argparse 

from covidnet import CovidNet
from get_data import get_dataloader
from utils.utils import read_data_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='Resnet50',
                        help="Model architecture to be used for the experiment.")
    parser.add_argument("--train", type=bool, default=False,
                        help="True for training mode. False for testing.")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of each mini-batch.")
    parser.add_argument("--lr", type=float, default=0.00001,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--train_metadata", type=str, default='train_split_v3.txt',
                        help="Path of the train metadata")
    parser.add_argument("--test_metadata", type=str, default='test_split_v3.txt',
                        help="Path of the test metadata")
    parser.add_argument("--model_path", type=str, default='weights',
                        help="Path of the test metadata")
    parser.add_argument("--cm_path", type=str, default='plots',
                        help="Path for the confusion_matrix")
    args = parser.parse_args() 

    covidnet = CovidNet(args) #Initializing the tester.
    if args.train:
        train_loader = get_dataloader(read_data_path(args.train_metadata),
                                      batch_size=args.batch_size)
        covidnet.train(train_loader)

    print('Testing a list of [COVID-19, COVID-19, normal, normal] cases')
    samples = covidnet.sample_images(['COVID-19', 'COVID-19', 'normal', 'normal'],
                                      test_data_path=args.test_metadata,
                                      batch_size=args.batch_size)
    covidnet.predict(samples)

    print('\n Testing all data')
    test_loader = get_dataloader(read_data_path(args.test_metadata),
                                 batch_size=args.batch_size)
    labels, preds = covidnet.predict(test_loader)
    covidnet.plot_cm_matrix(labels, preds, path=args.cm_path)