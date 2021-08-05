#include <iostream>
#include <fstream>
 #include <cstring>
 #include<cmath>
 #include<string>
#include<cmath>
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char *argv[]){
    string file1 = argv[1];
    string file2 = argv[2];
    ifstream ifile1;
    ifstream ifile2;
    ifile1.open (file1);
    ifile2.open(file2);
    int N = 0;
    map <float, float> map1;
    map <float, float> map2;
    string word;
    vector<float> keys;
    while (ifile1 >> word)
    {
        N++;
        float f = stof(word);
        keys.push_back(f);
        ifile1>>word;
        float g = stof(word);
        map1[f]=g;
    }
    while (ifile2 >> word)
    {
        float f = stof(word);
        ifile2>>word;
        float g = stof(word);
        map2[f]=g;
    }
    float utility = 0;
    for (int i=0;i<N;i++)
    {
     utility += exp(-fabs(map1[keys[i]]-map2[keys[i]]));
    
    }
    utility = utility/N;
    cout<<utility<<endl;
 }

