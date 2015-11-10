


all: 
	g++ -g -o rasp raspiberryPi.cpp -I /usr/include/opencv `pkg-config opencv --libs`
 
      
