#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>
#include <pthread.h>
#include <string>
#include "opencv2/highgui.hpp"
#include <chrono>
using namespace std;
using namespace cv;


int queue_density_main( Mat image, string test_video){   //processing at 5 fps
    
  VideoCapture cap(test_video);
  Mat background = image;
  cvtColor(background,background, COLOR_BGR2GRAY);
  Mat previous = background;
 
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  
  ofstream outfile;
  outfile.open("outbase1.txt");
  int count = 0;
   
  while(true){
    
    Mat frame;
    Size size(328,778);
    Mat dest = Mat::zeros(size,CV_8UC3);
    vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width-1, size.height -1), Point2f(0, size.height - 1 )};
   
    cap >> frame;
    
   
    if (frame.empty())
      break;
    
    vector<Point2f> data;
    data.push_back(Point2f(963,273));
    data.push_back(Point2f(1295,263));
    data.push_back(Point2f(1533,1007));
    data.push_back(Point2f(443,1006));
    
    
    Mat h = findHomography(data, points_dest);
   
    warpPerspective(frame, dest, h, size);
   
   cvtColor(dest, dest, COLOR_BGR2GRAY);
   
    Mat out;
    
    absdiff(dest,background,out);
   
    threshold(out,out,50,255,THRESH_BINARY);
   
    //imshow( "Frame1", out );
    
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    //cout<<contours.size()<<endl;
    count++;
   //outfile<<count<<" "<<contours.size()/1000.0<<" "<<contours0.size()/500.0<<endl;
    outfile<<double(count)/5<<" "<<contours.size()/1000.0<<endl;
    
    char c=(char)waitKey(10);
    if(c==27)
      break;
    cap>> frame;
    cap >> frame;
  }

  
  cap.release();
  destroyAllWindows();
  outfile.close();

  return 0;
}

int Method1( Mat image,int N,  string test_video){  //processing at 5 fps
 
  VideoCapture cap(test_video);
  
  Mat background = image;
  cvtColor(background,background, COLOR_BGR2GRAY);
  Mat previous = background;
 
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  
  ofstream outfile;
  outfile.open("output1.txt");
  int count = 0;
  int i =0;
  int k=0;
 
  while(true){
    
    Mat frame;
    Size size(328,778);
    Mat dest = Mat::zeros(size,CV_8UC3);
    vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width-1, size.height -1), Point2f(0, size.height - 1 )};
   
    cap >> frame;
    
    
    if (frame.empty())
      break;
    
    
    if(count==i*N){
        vector<Point2f> data;
        data.push_back(Point2f(963,273));
        data.push_back(Point2f(1295,263));
        data.push_back(Point2f(1533,1007));
        data.push_back(Point2f(443,1006));
        
        
        Mat h = findHomography(data, points_dest);
       
        warpPerspective(frame, dest, h, size);
       
       cvtColor(dest, dest, COLOR_BGR2GRAY);
       
        Mat out;
        
        absdiff(dest,background,out);
       
        threshold(out,out,50,255,THRESH_BINARY);
              
        
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        k =contours.size();
        
        outfile<<double(count)/5<<" "<<contours.size()/1000.0<<endl;
        i++;
        char c=(char)waitKey(10);
        if(c==27)
          break;
      }
    else{
       outfile<<double(count)/5<<" "<<k/1000.0<<endl;
    }

          count++;
          cap>>frame;
          cap>>frame;
    
  }

  
  cap.release();
  destroyAllWindows();
  outfile.close();

  return 0;
}


int Method2( Mat image, int x, int y,  string test_video){ //processing at 5 fps
  VideoCapture cap(test_video);
  
  Mat background = image;
  cvtColor(background,background, COLOR_BGR2GRAY);
  Mat previous = background;
 
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  
  ofstream outfile;
  outfile.open("output2.txt");
  int count = 0;
   
 
  while(true){
    
    Mat frame;
    Size size(328,778);
    Mat dest = Mat::zeros(size,CV_8UC3);
    vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width-1, size.height -1), Point2f(0, size.height - 1 )};
   
    cap >> frame;
    
    if (frame.empty())
      break;
    

   
    vector<Point2f> data;
    data.push_back(Point2f(963,273));
    data.push_back(Point2f(1295,263));
    data.push_back(Point2f(1533,1007));
    data.push_back(Point2f(443,1006));
    
    
    Mat h = findHomography(data, points_dest);
   
    warpPerspective(frame, dest, h, size);
   
   cvtColor(dest, dest, COLOR_BGR2GRAY);
      resize(dest, dest, Size(x, y), 0, 0, INTER_CUBIC);
   
    Mat out;
     resize(background, background, Size(x, y), 0, 0, INTER_CUBIC);
    absdiff(dest,background,out);
   
    threshold(out,out,50,255,THRESH_BINARY);    
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    count++;
    outfile<<double(count)/5<<" "<<contours.size()/1000.0<<endl;
    
    char c=(char)waitKey(10);
    if(c==27)
      break;
      
      cap>> frame;
      cap>> frame;
  }

  
  cap.release();
  destroyAllWindows();
  outfile.close();

  return 0;
}


struct structure_of_process_frame{
    double queue_density;
    Mat frame,background;
    int width,height,pos;
};

void *Process_Frame(void *structure_of_process_frame_object ){
    
 structure_of_process_frame * my_structure = (structure_of_process_frame * )structure_of_process_frame_object;
 Mat frame = my_structure->frame;
 Mat background = my_structure -> background;
 Mat out;
 
    absdiff(frame,background,out);
   
    threshold(out,out,50,255,THRESH_BINARY);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    my_structure -> queue_density = contours.size();
    
    pthread_exit(NULL);
}
 
int  Method3( Mat image, int N,  string test_video){ //processing at 5 fps
   
    VideoCapture cap(test_video);
    Mat background = image;
    cvtColor(background,background, COLOR_BGR2GRAY);//background is gray
   
    if(!cap.isOpened()){
      cout << "Error opening video stream or file" << endl;
       return -1;
    }
    
    ofstream outfile;
    outfile.open("output3.txt");
    
    int count = 0;
    //Mat frame0;
    Size size(328,778);
    Mat dest = Mat::zeros(size,CV_8UC3);
    vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width-1, size.height -1), Point2f(0, size.height - 1 )};
     
   
     
    Mat frame;
    
       
    Mat f[N];
    
    int w = background.size().width;
    int height = background.size().height;
    int temp = 0;
    
    
    for(int i=0;i<N ;i++){
            f[i]=background(Rect(0,temp,w,height/N));
            temp = temp + height/N;
        }
    
    while(true){
        
        pthread_t threads[N];
        structure_of_process_frame my_structure[N];
      
      
      Size size(328,778);
      Mat dest = Mat::zeros(size,CV_8UC3);
      vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width-1, size.height -1), Point2f(0, size.height - 1 )};
     
      cap >> frame;
      
     // resize(frame, frame, Size(200, 600), 0, 0, INTER_CUBIC);
      if (frame.empty())
        break;
      
       
     
      vector<Point2f> data;
      data.push_back(Point2f(963,273));
      data.push_back(Point2f(1295,263));
      data.push_back(Point2f(1533,1007));
      data.push_back(Point2f(443,1006));
        int rc ;
      
      Mat h = findHomography(data, points_dest);
     
      warpPerspective(frame, frame, h, size);
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        int width = frame.size().width;
        int height = frame.size().height;
        int t=0;
        for(int i =0;i<N;i++){
            my_structure[i].frame = frame(Rect(0,t,width,height/N));
            my_structure[i].height = height/N;
            my_structure[i].width = width;
            my_structure[i].pos =temp;
            my_structure[i].background = f[i];
            t = t + height/N;
            rc = pthread_create(&threads[i],NULL,Process_Frame,(void*)&my_structure[i]);
            if(rc){
                cout<<"Thread cant be created";
                break;
            }
        }
        double queue_density = 0.0;
        for(int i =0;i<N;i++){
            queue_density = queue_density+ my_structure[i].queue_density;
        }
        count++;
            //cout << count*3-2 << " " << queue_density/1000.0<< endl;
            outfile << double(count)/5 << " " << queue_density /1000.0<< endl;
          cap >>frame;
          cap>> frame;
        

    }

    
    cap.release();
    destroyAllWindows();
    outfile.close();

    return 0;
      
}

struct structure_of_process_frame2{
  string file;
  Mat background;
  int thread_num ,total_thread ;
  ofstream *outfile;
};

void *Process_(void *object2){
    
    
    structure_of_process_frame2 * my_structure2 = (structure_of_process_frame2 * )object2;
    string file = my_structure2 ->file;
    Mat background =my_structure2 -> background;
    int thread_num = my_structure2 -> thread_num;
    int total_thread= my_structure2 -> total_thread;
    // *(my_structure2->outfile);
    Mat frame;
    VideoCapture cap(file);
 //   Mat background = background;
    cvtColor(background,background, COLOR_BGR2GRAY);
    
   
    if(!cap.isOpened()){
      cout << "Error opening video stream or file" << endl;
     exit(0);
    }
    double queue_density;
   
    
    int count = 0;
    if(thread_num>0){
    for(int i =0;i<3*thread_num;i++ ){
        cap >> frame;
        count++;
        if (frame.empty())
          break;

    }}
     
    while(true){
      
      
      Size size(328,778);
      Mat dest = Mat::zeros(size,CV_8UC3);
      vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width-1, size.height -1), Point2f(0, size.height - 1 )};
     
      cap >> frame;
      
     
      if (frame.empty())
        break;
      
      vector<Point2f> data;
      data.push_back(Point2f(963,273));
      data.push_back(Point2f(1295,263));
      data.push_back(Point2f(1533,1007));
      data.push_back(Point2f(443,1006));
      
      
      Mat h = findHomography(data, points_dest);
     
      warpPerspective(frame, dest, h, size);
     
     cvtColor(dest, dest, COLOR_BGR2GRAY);
     
      Mat out;
      
      absdiff(dest,background,out);
     
      threshold(out,out,50,255,THRESH_BINARY);
        
      
      vector<vector<Point> > contours;
      vector<Vec4i> hierarchy;
      findContours( out, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
      queue_density=contours.size();
      count++;
        *(my_structure2->outfile) << double(count)/5 << " " << queue_density /1000.0<< endl;
        for(int i =0;i<3*total_thread -1;i++ ){
            cap >>frame;
            cap>>frame;
            cap>>frame;
            count++;
            if (frame.empty())
              break;

        }
        
            
      char c=(char)waitKey(10);
      if(c==27)
        break;
    }

    
    cap.release();
    destroyAllWindows();
   return 0;
}


void Method4(int N,Mat image,  string test_video){ //processing at 5fps
    int rc;
    ofstream outfile("output4.txt");
   
    pthread_t threads[N];
    structure_of_process_frame2 my_structure2[N];
  
    
    for(int i =0;i<N;i++){
        my_structure2[i].file = test_video;
        my_structure2[i].background = image;
        my_structure2[i].thread_num = i;
        my_structure2[i].total_thread = N;
        my_structure2[i].outfile = &outfile;

        rc = pthread_create(&threads[i],NULL,Process_,(void*)&my_structure2[i]);
        if(rc){
            cout<<"Thread cant be created";
            break;
        }
    }
    for( int i = 0; i < N; i++ ) {
        pthread_join(threads[i], NULL);
      }
    outfile.close();
    
}

Mat process(Mat image){
    Mat source = image;
    Size size = image.size();
   
    Mat dest = Mat::zeros(size,CV_8UC3);
    vector<Point2f> points_dest{Point2f(0,0), Point2f(size.width - 1, 0),Point2f(size.width - 1, size.height -1), Point2f(0, size.height - 1 )};
    vector<Point2f> points{Point2f(970,223), Point2f(1269,200),Point2f(1504,954), Point2f(309,951)};
    Mat h = findHomography(points, points_dest);
    warpPerspective(source, dest, h, size);
    return dest;

}

void method5(string method, string test_video) //extra credit problem, processing at 3 fps
{

  VideoCapture cap(test_video);
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return;
  }
  
  Mat background;
  cap>> background;
  cvtColor(background,background, COLOR_BGR2GRAY);
  Mat previous = background;
  previous = process(previous);
  ofstream outfile;
  outfile.open("pd.txt");
  int count = 0;
  
  if (method=="dense"){   //dense optical flow
  while(true){
      Mat frame, frame_gray;
      cap>> frame;
      
      if (frame.empty())
            break;
      count ++;
      frame= process(frame);
      cvtColor(frame,frame_gray, COLOR_BGR2GRAY);
      
      Mat flow(previous.size(), CV_32FC2);
      
      
      Mat mag, ang, mag_norm, flow_parts[2];
      calcOpticalFlowFarneback(previous, frame_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
      
      split(flow, flow_parts);
      
      cartToPolar(flow_parts[0], flow_parts[1], mag, ang, true);
      threshold(mag,mag,10,255,THRESH_BINARY);
      normalize(mag, mag_norm, 0.0f, 1.0f, NORM_MINMAX);
      ang *= ((1.f / 360.f) * (180.f / 255.f));
      outfile<<double(count)/3<<"\t"<<double(countNonZero(mag))/double(mag.rows)/double(mag.cols)<<endl;
      //cout<<count<<endl;
      previous = frame_gray;
      cap>> frame;
      cap>> frame;
      cap>> frame;
      cap>> frame;
  }

} else {   //sparse optical flow
   
   vector<Point2f> p0, p1;
   goodFeaturesToTrack(previous, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    while(true){
       Mat frame, frame_gray;
       cap>> frame;
       if (frame.empty())
            break;
       count ++;
       frame= process(frame);
       cvtColor(frame,frame_gray, COLOR_BGR2GRAY);
       vector<uchar> status;
       vector<float> err;
       TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
       calcOpticalFlowPyrLK(previous, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
       double dist =0;
       for(int i = 0; i < p0.size(); i++)
   {  
    dist += norm( p0[i]-p1[i]);
    }
    outfile<<double(count)/3<<"\t"<<dist*100/frame.rows/frame.cols<<endl;
    
      previous = frame_gray;
      cap >> frame;
      cap >> frame;
      cap>> frame;
      cap>> frame;
}}

} 
int main(int argc, char* argv[]){
    Mat image = imread(argv[1]);
    string test_video = argv[2];
 
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    //queue_density_main(image,  test_video);
    //Method1(image, 10, test_video );
    //Method2(image, 400, 600,  test_video);
    //Method3(image,10,  test_video);
    //method5("dense",  test_video);
    //method5("sparse",  test_video);
    Method4(10,image,  test_video);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_double.count()/1000 << " s"<<endl;    
}
