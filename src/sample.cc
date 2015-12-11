#include "general_settings.h"
#include <unordered_map>
#include "InputInit.h"

using namespace std;
using namespace cv;

std::vector<HiddenLayerConfig> hiddenConfig;
SoftmaxLayerConfig softmaxConfig;
std::vector<int> sample_vec;
///////////////////////////////////
// General parameters init before 
// reading config file
///////////////////////////////////
bool is_gradient_checking = false;
bool use_log = false;
int batch_size = 1;
int log_iter = 0;
int non_linearity = 2;
int training_epochs = 0;
//double lrate_w = 0.0;
//double lrate_b = 0.0;
float lrate_w = 0.0;
float lrate_b = 0.0;
int iter_per_epo = 0;
int word_vec_len = 0;
int nGram = 3;
float training_percent = 0.8;
int train_num=0;
void run()
{
    long start, end;
    start = clock();
    //读取配置文件
    readConfigFile("config.txt", true);

    std::vector<std::vector<singleWord> > trainData; //singleWord：key,word结构体
    std::vector<std::vector<singleWord> > testData;
    //unordered_map和map类似，都是存储的key-value的值，可以通过key快速索引到value。
    std::unordered_map<string, int> labelmap;//unordered_map：key,value形式 。label:分类的编号
    std::vector<string> re_labelmap;
    //读数据开始,labelmap.size(),re_labelmap.size()都是11，应该是类别
    readDataset("dataset/news_tagged_data.txt", trainData, testData, labelmap, re_labelmap);
    cout<<"Successfully read dataset, training data size is "<<trainData.size()<<", test data size is "<<testData.size()<<endl;
    softmaxConfig.NumClasses = labelmap.size();//11类，因为第二列就是10类
    /*
     * re_labelmap[i] ___PADDING___
	re_labelmap[i] B-NEWSTYPE
	re_labelmap[i] I-NEWSTYPE
	re_labelmap[i] O
	re_labelmap[i] B-KEYWORDS
	re_labelmap[i] I-KEYWORDS
	re_labelmap[i] B-PROVIDER
	re_labelmap[i] I-PROVIDER
	re_labelmap[i]
	re_labelmap[i] B-SECTION
	re_labelmap[i] I-SECTION
     */

    // change word 中的 number-word into "__DIGIT__"
    removeNumber(trainData);
    // get word map:
    // for each unique word in dataset, give it a unique id
    std::unordered_map<string, int> wordmap;
    std::vector<string> re_wordmap;
    //最后wordmap.size()和re_wordmap.size()都是1255，指每一个不同的word，作为一个key，有不同的value，从0开始的编号。
    //re_wordmap存放原始数据中不同的word,最后添加___UNDEFINED___，___PADDING___
    getWordMap(trainData, wordmap, re_wordmap);
    // For 1 of n encoding method, the input size of rnn is the size
    // of wordmap (one "1" and all others are "0")
    word_vec_len = re_wordmap.size();//1255

    std::vector<std::vector<int> > trainX;
    std::vector<std::vector<int> > trainY;
    // Break sentences into sub-sentences have length of nGram,
    // padding(填充) method is used
    //wordmap.size()是1255,指每一个不同的word，作为一个key，有不同的value，从0开始的编号。
    resolutioner(trainData, trainX, trainY, wordmap);//trainData是读每束加到向量中，有重复的键值
    cout<<"there are "<<trainX.size()<<" training data..."<<endl;//8852
    cout<<"there are "<<labelmap.size()<<" kind of labels..."<<endl;//11
    cout<<"there are "<<trainY.size()<<" kind of labels..."<<endl;//8852
    std::vector<std::vector<int> > testX;
    std::vector<std::vector<int> > testY;
    resolutioner(testData, testX, testY, wordmap);
    cout<<"there are "<<testX.size()<<" test data..."<<endl;//2249
    cout<<"there are "<<testY.size()<<" kind of labels of test data ..."<<endl;//2249

    train_num = trainX.size();//8852
    for(int i = 0; i < train_num; i++)
    {
        sample_vec.push_back(i);
    }
    std::vector<Hl> HiddenLayers;
    Smr smr;
    //初始化网络
    rnnInitPrarms(HiddenLayers, smr);
    Data2GPU(trainX, trainY, testX, testY,nGram);
    // Train network using Back Propogation（反向传导）
    //trainX：就是上面799束，将其每一束的5个组成一个句子，共8852组
    //trainY：8852组，HiddenLayers：隐含层，smr：softmax层；
    //testX：5个词组成一个句子，共2249组；testY：2249，共2249组；re_wordmap：r
    //re_wordmap存放原始数据中不同的word,最后添加___UNDEFINED___，___PADDING___,1255
    //trainX:0~1254      trainY:0~10
    trainNetwork(trainX, trainY, HiddenLayers, smr, testX, testY, re_wordmap);
    
    HiddenLayers.clear();
    trainData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);//c1.swap(c2):将c1和c2元素互换。
    testData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
}



int main(int argc, char** argv)
{

    run();
    return 0;
}




