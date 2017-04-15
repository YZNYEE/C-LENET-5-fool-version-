#ifndef CNN_NET_H_INCLUDED
#define CNN_NET_H_INCLUDED
#include"Matrix.h"

//CNN网
class CNN
{
public:
    CNN(int=0,int=0,int=0,int=0,int=0,int * =nullptr);
    //训练函数
    void Train(Matrix **,int,int,int batchsize,double Efficiency,Matrix **);
    //向前计算
    void FeedForward(Matrix &);
    //获得当前误差
    double GetError(Matrix &);
    //计算梯度
    void ComputeGradient(Matrix &,Matrix &);
    //累加梯度
    void ApplyGradient();
    //将累加梯度置零
    void SetZeros();
    //w=w-ThetaW
    void DGradient(int,double);
    //获取数据，只在我电脑上才行，
    static void GetExample(Matrix ***,Matrix ***,int);
    static void GetExample1(Matrix ***,Matrix ***,int);
    //测试函数
    void Test(Matrix **,Matrix **,int);
    //将CNN信息写入文件
    void WriteNetInFile();
    //从文件中创建CNN网
    static CNN * GetNetFromFile();
    //梯度检查原理  拉格朗日定理  (J(x+theta)-J(x-theta))/(2*theta) 约等于梯度
    //这个函数比较乱
    void CheckGradient(Matrix *,Matrix *);


    int inputmap; //图像的长度
    int NumOfC; //C层的数量
    int Kernel; //核的大小
    int NumOfS; //S层的数量
    int Scale;  //scale的大小
    int *NumMapOfC; //nummapofc[I]表示第i C层核的数量
    Matrix ***C;  //存卷积之后的矩阵  C[i][j]表示第i个卷积层，第j个矩阵
    Matrix ***CFeature; //存反向传导时矩阵，CFeature[i][j]表示第i个卷积层，第j个特征矩阵
                        //与C一一对应，不清楚的话，细谈
    Matrix ****CKernel; //存核，CKernel[i][j][k]第i个卷积层，第j个矩阵与上一层第k个矩阵的核
    Matrix ****CKGradient;//核的梯度，与核一一对应
    Matrix ****AllCKGradient;//一次训练核的总梯度，与核一一对应
    double **CB;        //CB[i][j]，表示第i个卷积层，第j个矩阵的偏度，一个实数，与C一一对应
    double **CBGradient;    //CB的梯度，与SB一一对应
    double **AllCBGradient;  //CB的总梯度
    Matrix ***S;        //存S层，S[I][J]懂？
    Matrix ***SFeature; //与S一一对应
    double **SW;        //SW[i][j]，S[I][J]矩阵的权值
    double **SWGradient;//SW[I][J]的梯度
    double **AllSWGradient;//总梯度
    double **SB;        //SB[I][J],S[I][J]矩阵的偏度
    double **SBGradient;//偏度的梯度
    double **AllSBGradient;
    Matrix *Vec;    //由最后一个矩阵层转化而成的向量，即特征提取完毕，常规神经网络运算
                    //倒数第三层
    Matrix *F;      //倒数第二层
    Matrix *WF;     //F层的权值
    Matrix *WFGradient;//梯度
    Matrix *AllWFGradient;//总梯度
    Matrix *Out;    //最后一层，即输出层
    Matrix *WOut;   //权值
    Matrix *WOutGradient;//梯度
    Matrix *AllWOutGradient;//总梯度

};

#endif // CNN_NET_H_INCLUDED
