#!/usr/bin/env python3

import argparse

import requests


def get_args():
    parser = argparse.ArgumentParser(description='Submit results.')
    parser.add_argument('-t', '--token', type=str, help='User Token', required=True)
    parser.add_argument('-f', '--file', type=str, help='Results file path', required=True)

    args = parser.parse_args()

    return args.token, args.file


if __name__ == '__main__':
    token, file = get_args()
    data = {
        'token': token
    }
    files = {
        'file': open(file, 'rb')
    }
    response = requests.post('http://79.175.132.14/visage/submit/', data=data, files=files)
    print('[{}] {}'.format(response.status_code, response.content.decode('ascii')))
