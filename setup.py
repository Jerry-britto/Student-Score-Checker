# responsible to create/use/deploy the ML application as a package
from setuptools import find_packages, setup
from typing import List

HYPERN_E_DOT='-e .'

def get_requirments(file_path:str) -> List[str]:
    '''
    This function will return a list of requirments
    '''
    requirments = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirments = [req.replace("\n"," ") for req in requirments]
        
    if HYPERN_E_DOT in requirments:
        requirments.remove(HYPERN_E_DOT)
    
    return requirments

setup(
    name="mlproject",
    version='0.0.1',
    author='Jerry',
    author_email='brittojerry1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirements.txt')
)