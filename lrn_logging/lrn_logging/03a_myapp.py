# 03a_myapp.py

import logging
import mylib


def main():
    logging.basicConfig(filename='03_myapp.log', level=logging.INFO)
    print('here')
    logging.info('Started')
    mylib.do_something()
    logging.info('Finished')


if __name__ == '__main__':
    main()
