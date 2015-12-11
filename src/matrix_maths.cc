#include "matrix_maths.h"

using namespace cv;
using namespace std;

double Reciprocal(const double &s){
    double res = 1.0;
    res /= s;
    return res;
}

Mat 
Reciprocal(const Mat &M){
    return 1.0 / M;
}


Mat 
sigmoid(const Mat &M){
    return 1.0 / (exp(-M) + 1.0);
}

Mat 
dsigmoid_a(const Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

Mat 
dsigmoid(const Mat &M){
    return divide(exp(M), pow((1.0 + exp(M)), 2.0));
}

Mat ReLUa(const Mat& M)
{
    Mat res = M > 0.0; //M中元素，大于0的元素 置为255 ， 小于等于0的 置0，得到res
    //在缩放或不缩放的情况下转换为另一种数据类型。
    //res – 目标矩阵。如果它的尺寸和类型不正确，在操作之前会重新分配。
    //CV_64FC1 – 要求是目标矩阵的类型，或者在当前通道数与源矩阵通道数相同的情况下的depth。
    //0 – 可选的delta加到缩放值中去。
    //该方法将源像素值转化为目标类型saturate_cast<> 要放在最后以避免溢出
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);//每个元素 × 1.0/255 + 0，故大于0 置为1 ， 其他 置0
    res = res.mul(M);//res矩阵除以M，大于0的还是原本的m值，其他为0
    return res;
}

Mat dReLUa(const Mat& M)
{
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255.0);
    return res;
}

Mat 
Tanh(const Mat &M){
    Mat res;
    M.copyTo(res);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            res.ATD(i, j) = tanh(M.ATD(i, j));
        }
    }
    return res;
}

Mat
dTanh(const Mat &M){
    Mat res = Mat::ones(M.rows, M.cols, CV_64FC1);
    return res - M.mul(M);
}

Mat 
nonLinearity(const Mat &M){
    if(non_linearity == NL_RELU){
        return ReLUa(M);
    }elif(non_linearity == NL_TANH){
        return Tanh(M);
    }else{
        return sigmoid(M);
    }
}

Mat 
dnonLinearity(const Mat &M){
    if(non_linearity == NL_RELU){
        return dReLUa(M);
    }elif(non_linearity == NL_TANH){
        return dTanh(M);
    }else{
        return dsigmoid(M);
    }
}

// Mimic rot90() in Matlab/GNU Octave.
Mat 
rot90(const Mat &M, int k){
    Mat res;
    if(k == 0) return M;
    elif(k == 1){
        flip(M.t(), res, 0);
    }else{
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}

// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat
kron(const Mat &a, const Mat &b){
    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC1);
    for(int i=0; i<a.rows; i++){
        for(int j=0; j<a.cols; j++){
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            Mat c = b.mul(a.ATD(i, j));
            c.copyTo(temp);
        }
    }
    return res;
}

Mat 
getBernoulliMatrix(int height, int width, double prob){
    // randu builds a Uniformly distributed matrix
    Mat ran = Mat::zeros(height, width, CV_64FC1);
    randu(ran, Scalar(0), Scalar(1.0));
    Mat res = ran <= prob;
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
    ran.release();
    return res;
}

// Follows are OpenCV maths
Mat 
exp(const Mat &src){
    Mat dst;
    exp(src, dst);
    return dst;
}

Mat 
log(const Mat &src){
    Mat dst;
    log(src, dst);
    return dst;
}

//M, 0, CV_REDUCE_MAX
Mat reduce(const Mat &src, int direc, int conf)//统计每列的最大元素
{
    Mat dst;
    //reduce(I,dst,int dim,int reduceOp,int dtype=-1);//可以统计每行或每列的最大、最小、平均值、和
    reduce(src, dst, direc, conf); // direc = 0 意味着矩阵被处理成1行 , direc = 1 意味着矩阵被处理成1列
    return dst;
}

//逐个元素相除，结果返回
Mat divide(const Mat &m1, const Mat &m2)
{
    Mat dst;
    divide(m1, m2, dst);
    return dst;
}

Mat 
pow(const Mat &m1, double val){
    Mat dst;
    pow(m1, val, dst);
    return dst;
}

double sum1(const Mat &m)
{
	//m.rows:11*50
	//tmp.rows:4*1
    Scalar tmp = sum(m);// 叠加每个通道的元素
    return tmp[0];//不知道返回的是什么
}

double
max(const Mat &m){
    Point min;
    Point max;
    double minval;
    double maxval;
    minMaxLoc(m, &minval, &maxval, &min, &max);
    return maxval;
}

double
min(const Mat &m){
    Point min;
    Point max;
    double minval;
    double maxval;
    minMaxLoc(m, &minval, &maxval, &min, &max);
    return minval;
}








