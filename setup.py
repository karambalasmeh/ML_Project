from setuptools import  find_packages,setup

dash_e_dot="-e ."
def get_req(file_path:str)->list[str]:
    """this function will return the list of requirements"""
    req=[]
    with open(file_path) as file_obj:
        req=file_obj.readlines()    
        req=[r.replace("\n","") for r in req]
        if dash_e_dot in req :
            req.remove(dash_e_dot)
    return req
setup(
    name="ML Project",
    version="0.1",
    author="karam",
    author_email="karam.balasmeh@gmail.com",
    packages=find_packages(),
    install_requires=get_req("requirements.txt")
    
    
    
    
)