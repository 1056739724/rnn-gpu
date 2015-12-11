#include "helper.h"
#include <unordered_map>

using namespace std;

// int to string
string i2str(int num){
    stringstream ss;
    ss<<num;
    string s = ss.str();
    return s;
}

// string to int
int str2i(string str){
    return atoi(str.c_str());
}

Mat 
vec2Mat(const std::vector<int> &labelvec){
    Mat res = Mat::zeros(1, labelvec.size(), CV_64FC1);
    for(int i = 0; i < labelvec.size(); i++){
//        res.ATD(0, i) = (double)(labelvec[i]);
    	res.ATD(0, i) = (float)(labelvec[i]);
    }
    return res;
}

bool 
isNumber(std::string &str){
    if(str.empty()) return false;
    for(int i = 0; i < str.size(); i++){
        if(str[i] < '0' || str[i] > '9') return false;
    }
    return true;
}

void removeNumber(std::vector<std::vector<singleWord> >& data)//训练集
{
    for(int i = 0; i < data.size(); i++)//799个句子，就是799束
    {
        for(int j = 0; j < data[i].size(); j++)
        {
            if(isNumber(data[i][j].word))//每一束的key，若是数字，则替代
            {
            	data[i][j].word = "___DIGIT___";
            }
        }
    }
}

void getWordMap(const std::vector<std::vector<singleWord> >& data,
		std::unordered_map<string, int> &map,
		std::vector<string> &re_map)
{
    map.clear();
    re_map.clear();
    for(int i = 0; i < data.size(); i++)//799
    {
        for(int j = 0; j < data[i].size(); j++)//某一束有多少对
        {
            if(map.find(data[i][j].word) == map.end())//map中没有找到
            {
                map[data[i][j].word] = re_map.size();//初始循环，这个re_map.size()值是0
                re_map.push_back(data[i][j].word);
            }
        }
    }
    map["___UNDEFINED___"] = re_map.size();
    re_map.push_back("___UNDEFINED___");
    map["___PADDING___"] = re_map.size();
    re_map.push_back("___PADDING___");//最后map.size()和re_map.size()都是1255，re_map存放map中的key,map存放不同键值对
}

int 
label2num(std::string label){
    int res = 0;
    if(label.compare("O") == 0){
        res = 0;
    }elif(label.compare("B-NEWSTYPE") == 0){
        res = 1;
    }elif(label.compare("B-PROVIDER") == 0){
        res = 2;
    }elif(label.compare("B-KEYWORDS") == 0){
        res = 3;
    }elif(label.compare("B-SECTION") == 0){
        res = 4;
    }elif(label.compare("I-NEWSTYPE") == 0){
        res = 5;
    }elif(label.compare("I-PROVIDER") == 0){
        res = 6;
    }elif(label.compare("I-KEYWORDS") == 0){
        res = 7;
    }elif(label.compare("I-SECTION") == 0){
        res = 8;
    }else{
        res = 9;
    }
    return res;
}

string 
num2label(int num){
    string res = "";
    if(num == 0){
        res = "O";
    }elif(num == 1){
        res = "B-NEWSTYPE";
    }elif(num == 2){
        res = "B-PROVIDER";
    }elif(num == 3){
        res = "B-KEYWORDS";
    }elif(num == 4){
        res = "B-SECTION";
    }elif(num == 5){
        res = "I-NEWSTYPE";
    }elif(num == 6){
        res = "I-PROVIDER";
    }elif(num == 7){
        res = "I-KEYWORDS";
    }elif(num == 8){
        res = "I-SECTION";
    }elif(num == 9){
        res = "ERROR";
    }
    return res;
}

void 
breakString(string str, std::vector<string> &vec){
    vec.clear();
    int head = 0;
    int tail = 0;
    while(true){
        if(head >= str.length()) break;
        if(str[tail] == ','){
            vec.push_back(str.substr(head, tail - head));
            head = tail + 1;
            tail = head;
        }else ++ tail;
    }
}

//one：0到1255中的某一个值；  n：1255
Mat oneOfN(int one, int n)
{
    Mat res = Mat::zeros(n, 1, CV_64FC1);//1225行1列
    res.ATD(one, 0) = 1.0;//返回对指定数组元素的引用。理解是对矩阵的one行0列赋值
    return res;
}



void getSample(const std::vector<std::vector<int> >& src1,//训练集子句子的组数，8852组，子句子的长度是5；范围：0～1254
		std::vector<Mat>& dst1,
		const std::vector<std::vector<int> >& src2,//8852组，原始数据第二列对应的int，表示类别，范围：0～10
		Mat& dst2,//行数是nGram 5,宽度是batch_size 50
		std::vector<string> &re_wordmap)////re_wordmap：存放原始数据中不同的word,最后添加___UNDEFINED___，___PADDING___,1255
{
    dst1.clear();
    int _size = dst2.cols; //batch_size,50
    int T = src1[0].size();//5
    printf("T:%d\n",T);
    for(int i = 0; i < T; i++)//5
    {
        Mat tmp = Mat::zeros(re_wordmap.size(), _size, CV_64FC1);//1列1样本；re_wordmap.size()是1255,_size是50指50个样本,1255*50
        dst1.push_back(tmp); //5个矩阵， 每个矩阵的相同列组成一个子句子
    }
    //random_shuffle()算法将序列中的元素进行乱序。该算法需要序列的起点迭代器和终点迭代器来进行乱序操作。
    //sample_vec：vector<int>是8852个子句子，里面存放索引，现在打乱
    random_shuffle(sample_vec.begin(), sample_vec.end());//sample_vec存放0-8851的序号，这边随机排序
    for(int i = 0; i < _size; i++)//50
    {//随机找50个子句子？
        int randomNum = sample_vec[i];//第randomNum个子句子
        for(int j = 0; j < T; j++)//5，每个子句子5行组成
        {
        	//src1[randomNum][j]：0到1255中的某一个值，是原始第一行对应的一个int值，；re_wordmap.size()：1255
            Mat tmp1 = oneOfN(src1[randomNum][j], re_wordmap.size());
            //Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
            //创建一个矩形对象，通过使用四个整数来初始化矩形左上角的横坐标、纵坐标以及矩形的宽度、高度（不要弄反）
            Rect roi = Rect(i, 0, 1, re_wordmap.size());//高度re_wordmap.size()
            Mat tmp2 = dst1[j](roi);//1255*1
            tmp1.copyTo(tmp2);//拷贝，将tmp1拷贝至tmp2,dst1已经赋值，不再是全0
//            Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
            dst2.ATD(j, i) = src2[randomNum][j];//实质是赋值
        }
//        dst2.ATD(0, i) = src2[randomNum][T - 1];
    }
}

//void cuda_getSample(const std::vector<std::vector<int> >& src1,//训练集子句子的组数，8852组，子句子的长度是5；范围：0～1254
//		cuMatrixVector& cuda_dst1,
//		const std::vector<std::vector<int> >& src2,//8852组，原始数据第二列对应的int，表示类别，范围：0～10
//		cuMatrix& cuda_dst2,//行数是nGram 5,宽度是batch_size 50
//		std::vector<string> &re_wordmap)////re_wordmap：存放原始数据中不同的word,最后添加___UNDEFINED___，___PADDING___,1255
//{
////    dst1.clear();
//	cuda_dst1.clear();
////    int _size = dst2.cols; //batch_size,50
//	int _size = cuda_dst2.cols();
//    int T = src1[0].size();//5
//    printf("_size:%d  T:%d\n",_size,T);
//    //下面的已经加了
////    for(int i = 0; i < T; i++)//5
////    {
////        Mat tmp = Mat::zeros(re_wordmap.size(), _size, CV_64FC1);//1列1样本；re_wordmap.size()是1255,_size是50指50个样本,1255*50
////        dst1.push_back(tmp); //5个矩阵， 每个矩阵的相同列组成一个子句子
////    }
//    //random_shuffle()算法将序列中的元素进行乱序。该算法需要序列的起点迭代器和终点迭代器来进行乱序操作。
//    //sample_vec：vector<int>是8852个子句子，里面存放索引，现在打乱
//    random_shuffle(sample_vec.begin(), sample_vec.end());//sample_vec存放0-8851的序号，这边随机排序
//    for(int i = 0; i < _size; i++)//50
//    {//随机找50个子句子？
//        int randomNum = sample_vec[i];//第randomNum个子句子
//        for(int j = 0; j < T; j++)//5，每个子句子5行组成
//        {
//        	//src1[randomNum][j]：0到1255中的某一个值，是原始第一行对应的一个int值，；re_wordmap.size()：1255
//            Mat tmp1 = oneOfN(src1[randomNum][j], re_wordmap.size());
//            //Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height);
//            //创建一个矩形对象，通过使用四个整数来初始化矩形左上角的横坐标、纵坐标以及矩形的宽度、高度（不要弄反）
//            Rect roi = Rect(i, 0, 1, re_wordmap.size());//高度re_wordmap.size()
//            Mat tmp2 = dst1[j](roi);//1255*1
//            tmp1.copyTo(tmp2);//拷贝，将tmp1拷贝至tmp2,dst1已经赋值，不再是全0
////            Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
//            dst2.ATD(j, i) = src2[randomNum][j];//实质是赋值
//        }
////        dst2.ATD(0, i) = src2[randomNum][T - 1];
//    }
//}

//void getDataMat(tmpBatch, sampleX, re_wordmap);
void 
getDataMat(const std::vector<std::vector<int> >& src, std::vector<Mat>& dst, std::vector<string> &re_wordmap){
    dst.clear();
    int _size = src.size();
    int T = src[0].size();
    for(int i = 0; i < T; i++){
        Mat tmp = Mat::zeros(re_wordmap.size(), _size, CV_64FC1);
        dst.push_back(tmp);
    }
    for(int i = 0; i < _size; i++){
        for(int j = 0; j < T; j++){
            Mat tmp1 = oneOfN(src[i][j], re_wordmap.size());
            Rect roi = Rect(i, 0, 1, re_wordmap.size());
            Mat tmp2 = dst[j](roi);
            tmp1.copyTo(tmp2);
        }
    }
}

void 
getLabelMat(const std::vector<std::vector<int> >& src, Mat& dst){
    int _size = dst.cols;
    int T = src[0].size();
    int mid = (int)(T /2.0);
    for(int i = 0; i < _size; i++){
        dst.ATD(0, i) = src[i][mid];
    }
}









