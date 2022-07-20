import requests
import os
from datetime import datetime
from tqdm import tqdm
import time

def download(url, pathname):
    now = datetime.now()
    date_time = now.strftime("%m%d%Y%H%M%S")
    # if path doesn't exist, make that path dir
    if not os.path.isdir(pathname):
        os.makedirs(pathname)
    # download the body of response by chunk, not immediately
    response = requests.get(url, stream=True)
    # get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # get the file name
    filename = os.path.join(pathname, date_time + ".jpg")
    # progress bar, changing the unit to bytes instead of iteration (default by tqdm)
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, "wb") as f:
        for data in progress.iterable:
            # write data read to the file
            f.write(data)
            # update the progress bar manually
            progress.update(len(data))

if __name__ == "__main__":
    path = os.getcwd()
    path = os.path.join(path, "goniopin", "pin_on")
    #path = os.path.join(path, "pin")
    while 1 > 0:
        download("http://bl23i-di-serv-02.diamond.ac.uk:8080/ECAM6.mjpg.jpg", path)
        time.sleep(1)
        #download("http://bl23i-di-serv-01.diamond.ac.uk:8080/ECAM9.mjpg.jpg", path)
        #time.sleep(1.1)
        #download("http://bl23i-di-serv-01.diamond.ac.uk:8080/ECAM10.mjpg.jpg", path)        
        #time.sleep(1.1)