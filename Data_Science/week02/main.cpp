#include <iostream>
#include <cstdio>
#include <sstream>
#include <string.h>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <set>
#include <thread>
#include <mutex>
#include <map>
#include <list>
#include <iomanip>
using namespace std;
typedef pair<short int, int>PAIR;
class Node{
public:
    short int item;
    int counts;
    Node* next_item = NULL; 
    vector<Node*> children;
    Node* parent = NULL;

    Node():counts(0), item(-1){}
    Node(short int itm):item(itm), counts(0){}
    Node(short int itm, int cnts):item(itm), counts(cnts){}
    Node(short int itm, int cnts, Node* next):item(itm), counts(cnts), next_item(next){}
    friend ostream &operator<<(ostream&, const Node&);
    ~Node(){}

};
ostream &operator<<(ostream &os, const Node& node){
    return os << "item = " << node.item << " counts = " << node.counts ;
}   
struct compBySet{
    bool operator()(const set<short int> &s1, const set<short int> &s2){
        set<short int>::iterator it_1 = s1.begin();
        set<short int>::iterator it_2 = s2.begin();
        if(s1.size() < s2.size())
            return true;
        else if (s1.size() > s2.size()){
            return false;
        }
        else{
            while(it_1 != s1.end()){
                if(*it_1 > *it_2)
                    return false;
                else if(*it_1 < *it_2)
                    return true;
                it_1++;
                it_2++;
            }
        }
    }
};

map<set<short int>, int, compBySet>all_combination;
map<set<short int>, int>::iterator iter;
vector< pair<short int, Node*> > header_table; 

int nme = 0;
bool pairCompare(const pair<short int, int>& a, const pair<short int, int>& b){
    if(a.second > b.second)
        return true;
    else if(a.second < b.second)
        return false; 
    return (a.first > b.first);
}
class FPTree{
public:
    map<vector<short int>, int> transaction_list;
    static map<short int, int>number_counts;
    int transaction_list_length;
    double min_support_line;
    
    Node* root;
    Node* record_header_link[1000];

    set<short int>base_combination;

public:
    FPTree(){}
    FPTree(double min_sp_line):min_support_line(min_sp_line){}
    FPTree(double min_sp_line, set<short int>base):min_support_line(min_sp_line),base_combination(base){}
    ~FPTree(){}

    static bool transactionCompare(const short int& a, const short int& b){
        if(number_counts[a] > number_counts[b])
            return true;
        else if(number_counts[a] < number_counts[b])
            return false;
        else{
            return (a>b);
        }
    }
        
    void fpGrowth(Node* node, set<short int> combination_set, int num){
        if (!node)
            return;
        //gMutex.lock();
        combination_set.insert(node->item);
        all_combination[combination_set] += num;
        //gMutex.unlock();
        
        while(node->parent->item != -1){
            fpGrowth(node->parent, combination_set, num);
            node = node->parent;
        }
        
    }
    
    void conditionPattern(Node* condition_node){
        header_table.clear();
        header_table.shrink_to_fit();
        number_counts.clear();

        // Create pattern conditional tree
        FPTree subtree(min_support_line, base_combination);
        bool tree_done = false;     

        // Add Header Table Line
        subtree.base_combination.insert(condition_node->item);
        while(condition_node){
            int c = condition_node->counts;
            all_combination[subtree.base_combination] += c;

            Node* tmp = condition_node->parent;
            vector<short int> transaction;
            tmp = condition_node->parent;
            while(tmp->parent){
                transaction.push_back(tmp->item);
                subtree.number_counts[tmp->item] += c;
                tmp = tmp ->parent;
            }
            subtree.transaction_list[transaction] += c;

            condition_node = condition_node->next_item;
        }

        subtree.buildTree();
        subtree.mineTree();
        deleteNode(subtree.root);

    }
    void mineTree(){
        // run all header table link to create subtree
        vector< pair<short int, Node*> > copy_header_table;
        copy_header_table.assign(header_table.begin(), header_table.end());
        for(short int i=copy_header_table.size()-1 ; i>=0 ;i-= 1){
            if (copy_header_table[i].second->next_item){
                if(copy_header_table[i].second->next_item->next_item)
                    conditionPattern(copy_header_table[i].second->next_item); 
                else{
                    fpGrowth(copy_header_table[i].second->next_item, base_combination, copy_header_table[i].second->next_item->counts);
                }
            }
        }  
    }        
    void saveFIle(char* output_file){ 
        ofstream  outfile(output_file);
        outfile.setf(ios::showpoint); 
        for(auto  it = all_combination.begin(); it != all_combination.end(); ++it)
        {
            for (short int const& i : it->first){
                if(i == *(it->first.begin()) )
                    outfile << i ;
                else
                    outfile <<","<< i ;
            }
            int c = (double)it->second/transaction_list_length * 1000000;
            if( c % 100 >= 45)
                c = c /100 + 1;
            else
                c = c/100;
            double d = (double)c/10000;
            outfile << fixed <<  setprecision(4) << ":"  << d << endl;
        }
    }
    void buildTree(){

        // Create header table
        // item counts, node link
        vector< pair<short int, int> > tmp_header;
        for(auto it = number_counts.begin(); it != number_counts.end(); ++it){
            if(it->second >= min_support_line){
                pair<short int, int> pr(it->first, it->second);
                tmp_header.push_back(pr);
            }
        }

        sort(tmp_header.begin(), tmp_header.end(), pairCompare);
        for(short int i=0;i<1000;i++){
            record_header_link[i] = new Node(-1, -1);
        }
        for(auto it = tmp_header.begin(); it != tmp_header.end(); ++it){
            pair<short int, Node*> pr(it->first, record_header_link[it->first]);
            header_table.push_back(pr);
        }
        

        root = new Node(-1, -1);
        for(auto  it = transaction_list.begin(); it != transaction_list.end(); ++it){
            vector<short int>trans = it->first;
            sort(trans.begin(), trans.end(), transactionCompare);
            short int end = trans.size()-1;
            while( end >=0 && number_counts[ trans[end] ] < min_support_line  ) {
                end --;
                trans.pop_back();
            }
            insert(trans, it->second);
        }        

    }
    void plotTree(string name){
        ofstream  outfile(name + ".txt");
        for(short int i=0;i<header_table.size();i++){
            Node* node = header_table[i].second->next_item;
            Node* trace = header_table[i].second->next_item;
            while(node){
                if(node->children.size() == 0){
                    trace = node;
                    while(trace->parent){
                        outfile << trace->item << " | ";
                        trace = trace->parent;
                    }
                    outfile << endl;
                }                
                node = node->next_item;
            }
          
        }
    }
    void insert(std::vector<short int> transaction, int num){
        Node* now_node = root;
        short int index;
        short int i = 0, item = -1;

        while( i<transaction.size() ){
            item = transaction[i];
            index = findItem(now_node->children, item);
            //cout << index << endl;
            // Not found
            
            if(index == -1){
                //cout << item << " ";
                now_node -> children.push_back(new Node(item, num));

                record_header_link[item]->next_item = now_node -> children.back();

                now_node->children.back()->parent = now_node;
                now_node = now_node -> children.back();

                record_header_link[item] = now_node;
            }
            else{
                now_node = now_node->children[index];
                now_node->counts += num;
            }
             i++;

        }
        //cout << endl;
    }

    void deleteNode(Node* node){
        if (!node)
            return;
        for(short int i=0;i<node->children.size();i++){
            deleteNode(node->children[i]);
        }
        free(node);
    }

    int findItem(vector<Node*> children, short int target_item){
        /* 
        ==========================================================
            Find the position of target number in vector
        ==========================================================
        */ 
        for(short int i=0;i<children.size();i++){
            if(children[i]->item == target_item){
                return i;
            }
        }
        return -1;
    }

    void readFile(char* input_file, double min_sup){
        /* 
        ==========================================================
            Read file from input file
        ==========================================================
        */        
        ifstream fin(input_file);
        string input_string;
        string number_string;
        short int num;
        int counts = 0;
        // Read input file
        while(getline(fin, input_string)){
            vector<short int> transaction;
            counts ++;
            num = 0;
            for(int i=0;i<input_string.size();i++){
                if( input_string[i]>='0' && input_string[i]<='9'){
                    num = num*10 + (int)input_string[i]-48;
                }
                else{
                    transaction.push_back(num);
                    number_counts[num] ++;
                    num = 0;
                }
            }
            if(num!=0 || transaction.size()== 0){
                transaction.push_back(num);
                number_counts[num] ++;
            }
            
            transaction_list[transaction] += 1;
        }
        transaction_list_length = counts;
        //min_support_line = (int)(min_sup * double(transaction_size) )
        min_support_line = (double)transaction_list_length * min_sup ;
    }

};
map<short int, int> FPTree::number_counts;
int main(int argc, char* argv[]){
    clock_t t_start = clock();

    double min_support = atof(argv[1]);
    char* input_file = argv[2];
    char* output_file = argv[3];

    // Build FP Tree
    FPTree fptree;
    fptree.readFile(input_file, min_support);
    cout << "Read file done" << endl;
    fptree.buildTree();
    cout << "Build fp tree done" << endl;
    
    fptree.mineTree();
    cout << "Mine tree done" << endl;
    fptree.saveFIle(output_file);
    cout << "Save file done" << endl;
    

    // Find all combination and save file
    //fptree.saveFIle(output_file);
    
    cout << "time taken = "<< (double)(clock()-t_start)/CLOCKS_PER_SEC << endl;
    return 0;
}