from setuptools import find_packages,setup ##it will find __init__.py in every folder and then that folde becomes a package
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    """
    requirement_lst:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!='-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Aditya Bansal",
    author_email = "bansaladitya048@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements() 
)