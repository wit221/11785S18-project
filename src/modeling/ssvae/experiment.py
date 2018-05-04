"""
Takes in model and performs prediction on given label(s) with/without suuplements
"""

import argparse


def experiment():
    # read in model

    # read in aggregated data

    # pass data through model

    #zero-out supplements

    #pass data through model

    #compare




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SS-VAE Experiment')

    parser.add_argument('--labels', nargs='+', default=['BLBX...'],
                        help="Labels to test against")

    args = parser.parse_args()
    print(args.labels)
