from setuptools import find_packages, setup
from typing import List
hye='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open("requirements.txt") as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if hye in requirements:
            requirements.remove(hye)

    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Roneel',
    author_email='roneelgovekar@yahoo.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)