from typing import List
from setuptools import find_packages,setup

def get_requirements(file_path:str)->List[str]:
    """ it return a list of requirements
    """
    requirements = []
    hypen = '-e .'
    with open(file_path)  as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements ]
        if hypen in requirements:
            requirements.remove(hypen)
    return requirements
setup(
    name ='Image_classification',
    version = '0.0.1',
    author = 'seetha',
    author_email = 'akashay8179@gmail.com',
    packages =find_packages(),
    install_requires = get_requirements('requirements.txt')  
    )