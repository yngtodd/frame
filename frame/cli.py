import argh

from argh import arg
from frame.framenet import save_sentence_data


@arg('datapath', help='Path to Framenet data')
@arg('saveroot', help='Root path to save csv data')
def preprocess_framenet(datapath, saveroot) -> None:
    r"""Save framenet sentences as csv files"""
    save_sentence_data(datapath, saveroot)


def main():
    parser = argh.ArghParser()
    parser.add_commands([preprocess_framenet])
    parser.dispatch()


if __name__=='__main__':
    main()
