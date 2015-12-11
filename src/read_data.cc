#include "read_data.h"
#include <unordered_map>

using namespace std;
using namespace cv;


/////////////label map!!!!
void readDataset(std::string path,
    std::vector<std::vector<singleWord> >& trainData, 
    std::vector<std::vector<singleWord> >& testData, 
    std::unordered_map<string, int> &labelmap, 
    std::vector<string> &re_labelmap)
{
    std::vector<std::vector<singleWord> > data;
    ifstream infile(path);
    string line;
    std::vector<singleWord> sentence;
    labelmap["___PADDING___"] = 0;
    re_labelmap.push_back("___PADDING___"); //re_labelmap 用于存放被标记过的label
    int counter = 1;
    while (getline(infile, line))
    {
        if(line.empty() || line[0] == ' ')//每一束结束就是空行，就走这条语句
        {
            if(!sentence.empty())//sentence就是读取出来的单词对应int label
            {
                data.push_back(sentence);
                sentence.clear();
            }
        }
        else
        {
        	//line : breaking	B-NEWSTYPE
            istringstream iss(line);
            string tmpword;
            string tmplabel;
            iss >> tmpword >> tmplabel;
            //tmpword: breaking
            //tmplabel: B-NEWSTYPE
            //用find函数来定位数据出现位置，它返回的一个迭代器，当数据出现时，它返回数据所在位置的迭代器，
            //如果map中没有要查找的数据，它返回的迭代器等于end函数返回的迭代器
            if(labelmap.find(tmplabel) == labelmap.end())//没找到
            {
                labelmap[tmplabel] = counter++;  //new label，通过tmplabel来对应int类型的labei
                re_labelmap.push_back(tmplabel);
            }
            singleWord tmpsw(tmpword, labelmap[tmplabel]);//word 和上面对应的int标签
            sentence.push_back(tmpsw);//把word 和对应的label加进向量中
        }
    }
    if(!sentence.empty())
    {
        data.push_back(sentence);
        sentence.clear();
    }
    // random shuffle
    //random_shuffle()算法将序列中的元素进行乱序。该算法需要序列的起点迭代器和终点迭代器来进行乱序操作。
    //在本例中，程序将scores.begin()和scores.end()返回的迭代器传递给算法。
    //这两个迭代器表示需要对scores中的全部元素进行乱序操作。程序的执行结果是，scores包含了相同的分数，只是顺序不同。
    //最后程序显示分数，证明乱序成功。
    random_shuffle(data.begin(), data.end());
    // cross validation ，交叉验证？？
    // data.size() 999
    //training_percent 0.8
    int trainSize = (int)((float)data.size() * training_percent);//799
    for(int i = 0; i < data.size(); ++i)//999
    {
        if(i < trainSize)
        {
        	trainData.push_back(data[i]);//前799束是训练集
        }
        else
        {
        	testData.push_back(data[i]);//后200束是测试集
        }
    }
    data.clear();
    std::vector<std::vector<singleWord> >().swap(data);//释放内存
}

// read data from stdin
void readLine(std::vector<string> &str)
{
    string line;
    getline(cin, line);

    istringstream stm(line);
    string word;
    while(stm >> word)
    {
        str.push_back(word);
    }
}

//    resolutioner(trainData, trainX, trainY, wordmap);
void resolutioner(const std::vector<std::vector<singleWord> > &data,
		std::vector<std::vector<int> > &resol,
		std::vector<std::vector<int> > &labels,
		std::unordered_map<string, int> &wordmap)//wordmap是1255
{
    std::vector<singleWord> tmpvec;
    std::vector<int> tmpresol;
    std::vector<int> tmplabel;
    for(int i = 0; i < data.size(); i++)//799束
    {
        tmpvec.clear();
        tmpvec = data[i];//第i个束
        singleWord tmpsw("___PADDING___", 0);
        int len = (int)(nGram / 2);//nGram=5，配置文件中定义的，5个词组成一个子句子
        //效果就是每束的前面和后面都增加两行("___PADDING___", 0);据说是为了对齐，存放在矩阵中，便于划分子句子
        for(int j = 0; j < len; j++)//2
        {
            tmpvec.insert(tmpvec.begin(), tmpsw);//插入元素：在第tmpvec.begin()前面插入tmpsw;
            tmpvec.push_back(tmpsw);//尾部插入
        }
        //上面这个for循环作用是在tmpvec向量前增加两行("___PADDING___", 0) key,value对，
        //后面增加两行("___PADDING___", 0)
        //下面这个for循环，我理解是将tmpvec分词，tmpvec.size()是13，每nGram也就是5个分为一词，总共分为13-5+1组即9组
        ////tmpvec.size():13,13-5+1=9,13个词可以分成9组，每组5个词
        for(int j = 0; j < tmpvec.size() - nGram + 1; j++)
        {
            tmpresol.clear();
            tmplabel.clear();
            for(int k = 0; k < nGram; k++)//5
            {
            	//wordmap里面有1255个key,value对，wordmap["___PADDING___"]=1254,
            	//wordmap["___UNDEFINED___"]=1253;
                if(wordmap.find(tmpvec[j + k].word) == wordmap.end())//没找到，应该能够找到的，所以执行else
                {
                    tmpresol.push_back(wordmap["___UNDEFINED___"]);
                }
                else
                {
                	//tmpvec[j + k].word是“___PADDING___“，wordmap[tmpvec[j + k].word]应该是1254
                	//j + k就是想取到每一组每个元素，将这一束的9组的每一组5个分类int值都加入进向量tmpresol中
                    tmpresol.push_back(wordmap[tmpvec[j + k].word]);
                }
                ////j + k就是想取到每一组每个元素，将这一束的9组的每一组5个分类label值都加入进向量tmpresol中
                tmplabel.push_back(tmpvec[j + k].label);
            }
            //tmpresol.size() 5，resol.size() 1，labels.size() 1
            resol.push_back(tmpresol);
            labels.push_back(tmplabel);
        }
    }
}
