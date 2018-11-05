#include <time.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
bool contain(vector<int>random, int target){
    for(int i=0;i<random.size();i++){
        if(random[i] == target){
            return true;
        }
    }
    return false;
}
int main(){
    ofstream  outfile("sample5.in");

    srand(time(NULL));
    int line = 10;
    int item_range = 15;
    int each_line = 5;
    vector<int> random_num;
    random_num.resize(each_line);
    for(int i = 0 ; i<line ; i++){
        for(int g=0;g<each_line;g++){
            int x = rand()%item_range;
            while(contain(random_num, x)){
                x =  rand()%item_range;
            }
            random_num[g] = x;
        }
        std::sort (random_num.begin(), random_num.end());
        // wrtie file
        for(int g=0;g<each_line;g++){
            if(g == each_line-1){
                outfile << random_num[g] << endl;
            }
            else{
                outfile << random_num[g] << ",";
            }
            
        }
    }
    return 0;
}