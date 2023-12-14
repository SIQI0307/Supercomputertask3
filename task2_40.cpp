#include<iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <omp.h>

using namespace std;


// 定义问题的参数
const double Lx = 1.5;  // length
const double Ly = 2.4;  // width
const int M = 40;       // nums_x
const int N = 40;       // nums_y
const double xbegin = -0.2;
const double ybegin = 1.2;
const double h1 = Lx/M;
const double h2 = Ly/N;
const double epsilon = pow(max(h1,h2),2);

vector<double> zrn(N+1,0);
vector<vector<double> > A(M+1,zrn);
vector<vector<double> > B(M+1,zrn);
vector<vector<double> > F(M+1,zrn);

//int dd = 0;// jiequ_cishu

bool judge(double x,double y);

double f_ot_x(double x,double y);

double f_ot_y(double y);

//double f(double x,double y);

double get_A(int i,int j);

double get_B(int i,int j);

double get_F(int i,int j);

double get_r(int i,int j,vector<vector<double> > &w);

double get_a(int i,int j);

double get_b(int i,int j);

double get_f(int i,int j);

string myitoa(int n);


int main(int argc,char *argv[]) {
    //double hx = Lx / M;  // x方向上的单元长度  h1
    //double hy = Ly / N;  // y方向上的单元长度  h2

    int desired_num_threads = stoi(argv[1]);

    omp_set_num_threads(desired_num_threads);


    // 初始化网格节点
    vector<double> xNodes(M + 1);
    vector<double> yNodes(N + 1);
    for (int i = 0; i <= M ; i++) {
        xNodes[i] = i * h1 + xbegin;
        //cout<<xNodes[i]<<" ";
    }
    //cout<<endl;
    for (int j = 0; j <= N; j++) {
        yNodes[j] = - j * h2 + ybegin;

    }


    
    int k;
    int iterations = 10000000;
    double delta = 0.000001;
    vector<vector<double> > w(M+1, vector<double>(N+1, 0.0));
    vector<vector<double> > new_w(M+1, vector<double>(N+1, 0.0));
    vector<vector<double> > r(M+1, vector<double>(N+1, 0.0));
    vector<vector<double> > Ar(M+1, vector<double>(N+1, 0.0));

/*
    for(int i=1;i<M;i++){
        for(int j=1;j<N;j++){
            bool b1 = judge(xNodes[j],yNodes[i]);
            if(b1) w[i-1][j-1] =0;
        }
    }
*/

    for(int i=0;i<M+1;i++){
        for(int j=0;j<N+1;j++){
            A[i][j] = get_A(i,j);
            B[i][j] = get_B(i,j);
            F[i][j] = get_F(i,j);
        }
    }



    int dd = 0;// jiequ_cishu

    auto start = chrono::high_resolution_clock::now();

    

    for (k = 0;; k++) {


        double numerator = 0;
        double denominator = 0;
        //int dd = 0;// jiequ_cishu



        #pragma omp parallel for collapse(2)  reduction(+ : numerator,denominator)

        

        for (int i = 1; i < M ; i++) {
            for (int j = 1; j < N ; j++) {

                double r_current, r_up, r_down, r_left, r_right;
                // 计算当前位置的 r 值
                r_current = get_r(i, j, w);
                
                // 计算上下左右位置的 r 值
                r_up = get_r(i+1, j, w);
                r_down = get_r(i-1, j, w);
                r_left = get_r(i, j-1, w);
                r_right = get_r(i, j+1, w);
                
                // 使用临时变量进行计算
                double Ar_current = - (get_a(i+1,j) * (r_up - r_current) - get_a(i,j) * (r_current - r_down)) / (h1 * h1)
                                - (get_b(i,j+1) * (r_right - r_current) - get_b(i,j) * (r_current - r_left)) / (h2 * h2);
                
                // 使用临时变量进行计算
                numerator += r_current * Ar_current;
                denominator += Ar_current * Ar_current;
            }
        }

        
        
        //cout<<numerator<<" "<<denominator<<endl;

        double tau = numerator / denominator;
        double error = 0;

        #pragma omp parallel for collapse(2) reduction(max : error)

        
        for(int i=1;i<M;i++){                    
            for(int j=1;j<N;j++){
                double r = get_r(i,j,w);
                new_w[i][j] = w[i][j] - tau * r;
                error = max(error,fabs(tau * r));
            }
            
        }

        //#pragma omp parallel for collapse(2)

        for(int i=1;i<M;i++){                    
            for(int j=1;j<N;j++){
            	w[i][j] = new_w[i][j];
            }
            
        }

        //if(k<25) cout<<error<<endl;

        
        if(error < delta) break;



       
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    double time = duration.count();
    //cout<<time;
/*
    for(int i=0;i<M-1;i++){
        for(int j=0;j<N-1;j++)
            printf("%6.3f ",w[i][j]);
        cout<<endl;
    }
*/
    //cout<<k<<endl; // 迭代次数
    //cout<<M<<endl;
    //cout<<N<<endl;
    
    //cout<<M<<" "<<N;
    //cout<<"finished"<<endl;
    
    cout << k <<endl;
    cout << M <<endl;
    cout << N <<endl;
    cout << time <<endl;
    cout << desired_num_threads << endl;
    
    for(int i=0;i<M+1;i++)
        cout << xNodes[i] << " ";
    cout << endl;

    for(int i=0;i<N+1;i++)
        cout << yNodes[i] << " ";
    cout << endl;
    

    for(int i=0;i<M-1;i++){
        for(int j=0;j<N-1;j++)
            cout << w[i][j] << " ";
        cout << endl;
    }
    
    return 0;
}


double f_ot_x(double x,double y){
    if (x<=0) return 0;
    if(y > 0) return pow(x,0.5);
    else if(y < 0) return -pow(x,0.5);
    else return 0;
}

double f_ot_y(double y){
    return pow(y,2); // y*y
}

bool judge(double x,double y){
    return x>0 && y*y < x && x<1 ; 
}





double get_A(int i,int j){

    if(i<=0 || i>=M || j<=0 || j>=N) return 1/epsilon;
    double x = j * h1 + xbegin;
    double y = - i * h2 + ybegin;
    double y1 = - (i-1) * h2 + ybegin;
    double y5 = - (i+1) * h2 + ybegin;

    bool jg1 = judge(x,y1);
    bool jg2 = judge(x,y5);
    bool jg = judge(x,y);

    if(jg1 && jg2) return 1;
    else if(!jg1 && !jg2){
        if(!jg) return 1/epsilon;
        else{
            double ll = f_ot_x(x,1);
            if(ll > y1) return 1;
            else return 2*ll/h2 + (1-2*ll/h2)/epsilon;
        }
    }

    else{
        double yy = f_ot_x(x,y);
        double y2 = (y1 + y) / 2;
        double y4 = (y + y5) / 2;

        if(judge(x,y5)){
            if(yy >= y2) return 1;
            else if(yy <=y4) return 1/epsilon;
            else return (yy-y4)/h2 + (1-(yy-y4)/h2)/epsilon;
        }
        else{
            if(yy >= y2) return 1/epsilon;
            else if(yy <= y4) return 1;
            else return (y2-yy)/h2 + (1-(y2-yy)/h2)/epsilon;
        }
    } 

    return 1;



}

double get_B(int i,int j){
    if(i<=0 || i>=M || j<=0 || j>=N) return 1/epsilon;

    double x = j * h1 + xbegin;
    double y = - i * h2 + ybegin;
    double x1 = (j-1) * h1 + xbegin;
    double x5 = (j+1) * h1 + xbegin;

    bool jg1 = judge(x1,y);
    bool jg2 = judge(x5,x);
    bool jg = judge(x,y);

    if(jg1 && jg2) return 1;
    else if(!jg1 && !jg2 && !jg) return 1/epsilon;
    else if(!jg2){
        double xx = f_ot_y(y);
        double x2 = (x1 + x)/2;
        double x4 = (x + x5)/2;
        if(x2 > 1) return 1/epsilon;
        else if(xx > x4 ) return 1/epsilon;
        else{
            double ll = min(x4,1.0)-max(x2,xx);
            return ll/h1 + (1-ll/h1)/epsilon;
        }        
    }

    else{
        double xx = f_ot_y(y);
        double x2 = (x1 + x)/2;
        double x4 = (x + x5)/2;
        //cout<<" y1:"<<yNodes[i-1]<<" yy:"<<yy<<" y5:"<<yNodes[i+1]<<endl;

        if(xx >= x4) return 1/epsilon;
        else if(xx <= x2) return 1;
        else return (x4-xx)/h1 + (1-(x4-xx)/h1)/epsilon;
    } 

    return 1;
}

double get_F(int i,int j){  // bushi  i-1 j-1
    
    double x = j * h1 + xbegin;
    double y = - i * h2 + ybegin;
    double x1 = x - h1/2, x2 = x + h1/2, y1 = y + h2/2, y2 =y - h2/2;
        
    bool b1 = judge(x1, y1);
    bool b2 = judge(x1, y2);
    bool b3 = judge(x2, y1);
    bool b4 = judge(x2, y2);

    if(b1 && b2 && b3 && b4) return 1;
        
    else{
        double nn = 0;
        double xd = (x2-x1)/24;
        double yd = (y2-y1)/24;
        for(int k=0;k<25;k++){
            for(int l=0;l<25;l++){
                if(judge(x1+k*xd,y1+l*yd)) nn++;
            }
        }
        return nn/625;
    }
    return 1;
}

double get_r(int i,int j,vector<vector<double> > &w){
    
    if(i<=0 || i>=M || j<=0 || j>=N) return 0;
    
    /*
    return  - (get_A(i+1,j) * (w[i+1][j] - w[i][j]) - get_A(i,j) * (w[i][j] - w[i-1][j]) ) / (h1 * h1)
            - (get_B(i,j+1) * (w[i][j+1] - w[i][j])  - get_B(i,j) * (w[i][j] - w[i][j-1]) ) / (h2 * h2)
            - get_F(i,j); 
    */
       return  - (get_a(i+1,j) * (w[i+1][j] - w[i][j]) - get_a(i,j) * (w[i][j] - w[i-1][j]) ) / (h1 * h1)
            - (get_b(i,j+1) * (w[i][j+1] - w[i][j])  - get_b(i,j) * (w[i][j] - w[i][j-1]) ) / (h2 * h2)
            - get_f(i,j); 
}

string myitoa(int n){
    int x = n;
    int i = 0;
    string s;
    while(1){
        if(x /10 != 0) {
            s.push_back(x % 10 + '0');
            x /= 10;
        }
        else{
            s.push_back(x % 10 + '0');
            break;
        }
    }

    string ss;
    for(int i=s.size()-1;i>=0;i--)
        ss.push_back(s[i]);


    return ss;


}

double get_a(int i,int j){
    return A[i][j];
}

double get_b(int i,int j){
    return B[i][j];
}

double get_f(int i,int j){
    return F[i][j];
}


/*
    for(int i=0;i<=M;i++){
        for(int j=0;j<=N;j++){
            if(judgement[i][j]) cout<<1<<" ";
            else cout<<0<<" ";
        }
        cout<<endl;
    }
*/




/*
        if((k+1) == pow(10,dd)){
            dd++;
            string s;
            s += "output";
            s += myitoa(M);
            s += "_";
            s += myitoa(N);
            s += "_k+1=";
            s += myitoa(k+1);
            s += ".txt";
            
            
            ofstream file1(s);

            file1 << k <<endl;
            file1 << M <<endl;
            file1 << N <<endl;
            file1 << time <<endl;
            
            for(int i=0;i<M+1;i++)
                file1 << xNodes[i] << " ";
            file1 << endl;

            for(int i=0;i<N+1;i++)
                file1 << yNodes[i] << " ";
            file1 << endl;
            

            for(int i=0;i<M-1;i++){
                for(int j=0;j<N-1;j++)
                    file1 << w[i][j] << " ";
                file1 << endl;
            }
        }


        //int dd = 0;// jiequ_cishu

*/

