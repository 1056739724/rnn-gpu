#include "read_config.h"

using namespace std;

string read_2_string(string File_name)
{
    char *pBuf;
    FILE *pFile = NULL;   
    //参数path字符串包含欲打开的文件路径及文件名，参数mode字符串则代表着流形态
    //r是只读方式，该文件必须存在
    //文件顺利打开后，指向该流的文件指针就会被返回。如果文件打开失败则返回NULL，并把错误代码存在errno中
    if(!(pFile = fopen(File_name.c_str(),"r")))
    {
        printf("Can not find this file.");
        return 0;
    }
    //move pointer to the end of the file
    //int fseek(FILE *stream, long offset, int fromwhere);函数设置文件指针stream的位置。
    //如果执行成功，stream将指向以fromwhere为基准，偏移offset（指针偏移量）个字节的位置，函数返回0。
    //如果执行失败(比如offset超过文件自身大小)，则不改变stream指向的位置，函数返回一个非0值。
    //超出文件末尾位置，还是返回0。往回偏移超出首位置，返回-1，且指向一个-1的位置，请小心使用。
    fseek(pFile, 0, SEEK_END);
    //Gets the current position of a file pointer.offset 
    //函数 ftell 用于得到文件位置指针当前位置相对于文件首的偏移字节数。在随机方式存取文件时，
    //由于文件位置频繁的前后移动，程序不容易确定文件的当前位置。
    int len = ftell(pFile);
    pBuf = new char[len];
    //Repositions the file pointer to the beginning of a file
    //rewind(pFile),C 程序中的库函数，功能是将文件内部的指针重新指向一个流的开头
    rewind(pFile);
    //size_t fread ( void *buffer, size_t size, size_t count, FILE *stream) ;
    //buffer,用于接收数据的内存地址;size,要读的每个数据项的字节数，单位是字节\
    //count,要读count个数据项，每个数据项size个字节.
    //stream,输入流
    //如果调用成功返回实际读取到的元素个数，如果不成功或读到文件末尾返回 0。
    fread(pBuf, 1, len, pFile);
    fclose(pFile);
    string res = pBuf;
    return res;
}

bool get_word_bool(string &str, string name)//str:配置文件  name：IS_GRADIENT_CHECKING
{
	//str：两段
    size_t pos = str.find(name);//0
    int i = pos + 1;
    bool res = true;
    while(1)
    {
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    //str:IS_GRADIENT_CHECKING 到 TRAINING_PERCENT=0.80;
    string sub = str.substr(pos, i - pos + 1);
    //sub:IS_GRADIENT_CHECKING=false;
    if(sub[sub.length() - 1] == ';')
    {
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);//content:false
        if(!content.compare("true")) res = true;
        else res = false;
    }
    str.erase(pos, i - pos + 1);//擦除IS_GRADIENT_CHECKING=false;这一句话，str：USE_LOG=false  到 TRAINING_PERCENT=0.80;
    return res;//0
}

int get_word_int(string &str, string name){

    size_t pos = str.find(name);    
    int i = pos + 1;
    int res = 1;
    while(1)
    {
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    string sub = str.substr(pos, i - pos + 1);
    if(sub[sub.length() - 1] == ';')
    {
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        res = stoi(content);
    }
    str.erase(pos, i - pos + 1);
    return res;
}

double get_word_double(string &str, string name)
{
    size_t pos = str.find(name);    
    int i = pos + 1;
    double res = 0.0;
    while(1)
    {
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    //str :ITER_PER_EPO=100;LRATE_W=3e-3;LRATE_B=1e-3;NGRAM=5;TRAINING_PERCENT=0.80;
    string sub = str.substr(pos, i - pos + 1);
    //sub :LRATE_W=3e-3;
    if(sub[sub.length() - 1] == ';')
    {
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        // content 3e-3
        //content.c_str() :3e-3
        res = atof(content.c_str());//atof:把字符串转换成浮点数名
        //res :0.003
    }
    str.erase(pos, i - pos + 1);

    return res;
}

int get_word_type(string &str, string name) //ayers[i], "LAYER"
{
    size_t pos = str.find(name); //find 函数 返回name在str中的下标位置，0
    int i = pos + 1;
    int res = 0;
    while(1)//每次读取一行
    {
        if(i == str.length()) break;
        if(str[i] == ';') break;
        ++ i;
    }
    //str:LAYER=HIDDEN;NUM_HIDDEN_NEURONS=512;WEIGHT_DECAY=1e-6;DROPOUT_RATE=1.0;
    string sub = str.substr(pos, i - pos + 1);
    //sub:LAYER=HIDDEN;
    if(sub[sub.length() - 1] == ';')
    {
    	//截取从LAYER后面到；前面的内容
        string content = sub.substr(name.length() + 1, sub.length() - name.length() - 2);
        //content:HIDDEN
        if(!content.compare("NL_SIGMOID")) res = 0;
        elif(!content.compare("NL_TANH")) res = 1;
        elif(!content.compare("NL_RELU")) res = 2;
        elif(!content.compare("HIDDEN")) res = 0;
        elif(!content.compare("SOFTMAX")) res = 1;
    }
    str.erase(pos, i - pos + 1);//擦除了LAYER = HIDDEN;
    return res;
}


void delete_comment(string &str)
{
    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1)
    {
        if(head == str.length()) break;
        if(str[head] == '/')
        {
            tail = head + 1;
            while(1)
            {
                if(tail == str.length()) break;
                if(str[tail] == '/') break;
                ++ tail;
            }
            str.erase(head, tail - head + 1);
        }else ++ head;
    }
}

void delete_space(string &str)
{
    if(str.empty()) return;
    int i = 0;
    while(1)//循环
    {
        if(i == str.length()) break;
        if(str[i] == '\t' || str[i] == '\n' || str[i] == ' ')//不等于tab按键
        {
            str.erase(str.begin() + i);//清除str.begin() + i指向的元素, 返回指向下一个字符的迭代器,
        }
        else
        {
        	++ i;
        }
    }
}

void get_layers_config(string &str)
{
    std::vector<string> layers;
    if(str.empty()) return;
    int head = 0;
    int tail = 0;
    while(1)//循环结束，配置文件中主要信息都加进vector中
    {
        if(head == str.length()) break;
        if(str[head] == '$')
        {
            tail = head + 1;
            while(1)
            {
                if(tail == str.length()) break;
                if(str[tail] == '&') break;
                ++ tail;
            }
            string sub = str.substr(head, tail - head + 1);//截取字符串，截取$到&之间
            if(sub[sub.length() - 1] == '&')
            {
                sub.erase(sub.begin() + sub.length() - 1);//擦除'&'
                sub.erase(sub.begin());//擦除'$'??
                layers.push_back(sub);//加入vector<string>中
            }
            //删除从start到end的所有字符, 返回一个迭代器,指向被删除的最后一个字符的下一个位置
            str.erase(head, tail - head + 1);
        }
        else
        {
        	++ head;
        }
    }
    for(int i = 0; i < layers.size(); i++)//向量中$和&之间的2个string，都是关于层的信息
    {
        int type = get_word_type(layers[i], "LAYER");
        switch(type)
        {
            case 0://HIDDEN
            {
                int hn = get_word_int(layers[i], "NUM_HIDDEN_NEURONS");//512
                double wd = get_word_double(layers[i], "WEIGHT_DECAY");//1e-6
                double dor = get_word_double(layers[i], "DROPOUT_RATE");//1.0
                hiddenConfig.push_back(HiddenLayerConfig(hn, wd, dor));
                break;
            }
            case 1://SOFTMAX
            {
                softmaxConfig.NumClasses = get_word_int(layers[i], "NUM_CLASSES");//0
                softmaxConfig.WeightDecay = get_word_double(layers[i], "WEIGHT_DECAY");//1e-6
                break;
            }
        }
    }
}

//读取配置文件，第一个参数：文件路径，第二个参数：showinfo值true，标识显不显示信息
void readConfigFile(string filepath, bool showinfo)
{
    string str = read_2_string(filepath);//文件内容读到str中
    delete_space(str);//清除空格
    delete_comment(str);//清除注释
    get_layers_config(str);//读取配置文件隐藏层和softmax层

    is_gradient_checking = get_word_bool(str, "IS_GRADIENT_CHECKING");//得到false
    
    if(is_gradient_checking)//测试求导是否正确is_gradient_checking
    {
        for(int i = 0; i < hiddenConfig.size(); i++)//值1，hiddenConfig是个向量，向量中是隐含层结构体,
        {
            hiddenConfig[i].DropoutRate = 1.0;
        }
    }
    //str：USE_LOG=false;NON_LINEARITY=NL_RELU;BATCH_SIZE=50;TRAINING_EPOCHS=30;
    //ITER_PER_EPO=100;LRATE_W=3e-3;LRATE_B=1e-3;NGRAM=5;TRAINING_PERCENT=0.80;
    use_log = get_word_bool(str, "USE_LOG");//use_log 0
    batch_size = get_word_int(str, "BATCH_SIZE");//batch_size 50
    non_linearity = get_word_type(str, "NON_LINEARITY");//non_linearity 2

    training_epochs = get_word_int(str, "TRAINING_EPOCHS");//training_epochs 30
    lrate_w = get_word_double(str, "LRATE_W");//lrate_w 0.003
    lrate_b = get_word_double(str, "LRATE_B");//lrate_b 0.001
    iter_per_epo = get_word_int(str, "ITER_PER_EPO");//iter_per_epo 100
    nGram = get_word_int(str, "NGRAM");//nGram 5
    training_percent = get_word_double(str, "TRAINING_PERCENT");//training_percent 0.8
    if(!showinfo) return;//showinfo:true,!showinfo:false
    cout<<"****************************************************************************"<<endl
        <<"**                    READ CONFIG FILE COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;

    for(int i = 0; i < hiddenConfig.size(); i++)//1
    {
        cout<<"***** hidden layer: "<<i<<" *****"<<endl;
        cout<<"NumHiddenNeurons = "<<hiddenConfig[i].NumHiddenNeurons<<endl;
        cout<<"WeightDecay = "<<hiddenConfig[i].WeightDecay<<endl;
        cout<<"DropoutRate = "<<hiddenConfig[i].DropoutRate<<endl<<endl;
    }
    cout<<"***** softmax layer: *****"<<endl;
//    cout<<"NumClasses = "<<softmaxConfig.NumClasses<<endl;
    cout<<"WeightDecay = "<<softmaxConfig.WeightDecay<<endl<<endl;//权重衰减

    cout<<"***** general config *****"<<endl;
    cout<<"is_gradient_checking = "<<is_gradient_checking<<endl;
    cout<<"use_log = "<<use_log<<endl;
    cout<<"batch size = "<<batch_size<<endl;

    cout<<"training epochs = "<<training_epochs<<endl;
    cout<<"learning rate for weight matrices = "<<lrate_w<<endl;
    cout<<"learning rate for bias = "<<lrate_b<<endl;
    cout<<"iteration per epoch = "<<iter_per_epo<<endl;
    cout<<"n gram = "<<nGram<<endl;
    cout<<"training percent = "<<training_percent<<endl;
    cout<<endl;
}
