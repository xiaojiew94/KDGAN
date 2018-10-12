### KDGAN

knowledge distillation generative adversarial network  

### Server

ssh xiaojie@10.100.229.246 # cpu   
ssh xiaojie@10.100.228.181 # gpu xw  
ssh xiaojie@10.100.228.158 # gpu cz 

### Problem
PermissionError: [Errno 13] Permission denied
sudo chmod -R ugo+rw /data


### YFCC100M

wget http://download.maxmind.com/download/worldcities/worldcitiespop.txt.gz
gunzip worldcitiespop.txt.gz
file -i worldcitiespop.txt

00 Line number  
01 Photo/video identifier  
02 Photo/video hash  
03 User NSID  
04 User nickname  
05 Date taken  
06 Date uploaded  
07 Capture device  
08 Title  
09 Description  
10 User tags (comma-separated)  
11 Machine tags (comma-separated)  
12 Longitude  
13 Latitude  
14 Accuracy of the longitude and latitude coordinates (1=world level accuracy)  
15 Photo/video page URL  
16 Photo/video download URL  
17 License name  
18 License URL  
19 Photo/video server identifier  
20 Photo/video farm identifier  
21 Photo/video secret  
22 Photo/video secret original  
23 Extension of the original photo  
24 Photos/video marker (0=photo, 1=video)  



