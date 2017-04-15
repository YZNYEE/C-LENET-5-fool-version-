#ifndef CNN_NET_H_INCLUDED
#define CNN_NET_H_INCLUDED
#include"Matrix.h"

//CNN��
class CNN
{
public:
    CNN(int=0,int=0,int=0,int=0,int=0,int * =nullptr);
    //ѵ������
    void Train(Matrix **,int,int,int batchsize,double Efficiency,Matrix **);
    //��ǰ����
    void FeedForward(Matrix &);
    //��õ�ǰ���
    double GetError(Matrix &);
    //�����ݶ�
    void ComputeGradient(Matrix &,Matrix &);
    //�ۼ��ݶ�
    void ApplyGradient();
    //���ۼ��ݶ�����
    void SetZeros();
    //w=w-ThetaW
    void DGradient(int,double);
    //��ȡ���ݣ�ֻ���ҵ����ϲ��У�
    static void GetExample(Matrix ***,Matrix ***,int);
    static void GetExample1(Matrix ***,Matrix ***,int);
    //���Ժ���
    void Test(Matrix **,Matrix **,int);
    //��CNN��Ϣд���ļ�
    void WriteNetInFile();
    //���ļ��д���CNN��
    static CNN * GetNetFromFile();
    //�ݶȼ��ԭ��  �������ն���  (J(x+theta)-J(x-theta))/(2*theta) Լ�����ݶ�
    //��������Ƚ���
    void CheckGradient(Matrix *,Matrix *);


    int inputmap; //ͼ��ĳ���
    int NumOfC; //C�������
    int Kernel; //�˵Ĵ�С
    int NumOfS; //S�������
    int Scale;  //scale�Ĵ�С
    int *NumMapOfC; //nummapofc[I]��ʾ��i C��˵�����
    Matrix ***C;  //����֮��ľ���  C[i][j]��ʾ��i������㣬��j������
    Matrix ***CFeature; //�淴�򴫵�ʱ����CFeature[i][j]��ʾ��i������㣬��j����������
                        //��Cһһ��Ӧ��������Ļ���ϸ̸
    Matrix ****CKernel; //��ˣ�CKernel[i][j][k]��i������㣬��j����������һ���k������ĺ�
    Matrix ****CKGradient;//�˵��ݶȣ����һһ��Ӧ
    Matrix ****AllCKGradient;//һ��ѵ���˵����ݶȣ����һһ��Ӧ
    double **CB;        //CB[i][j]����ʾ��i������㣬��j�������ƫ�ȣ�һ��ʵ������Cһһ��Ӧ
    double **CBGradient;    //CB���ݶȣ���SBһһ��Ӧ
    double **AllCBGradient;  //CB�����ݶ�
    Matrix ***S;        //��S�㣬S[I][J]����
    Matrix ***SFeature; //��Sһһ��Ӧ
    double **SW;        //SW[i][j]��S[I][J]�����Ȩֵ
    double **SWGradient;//SW[I][J]���ݶ�
    double **AllSWGradient;//���ݶ�
    double **SB;        //SB[I][J],S[I][J]�����ƫ��
    double **SBGradient;//ƫ�ȵ��ݶ�
    double **AllSBGradient;
    Matrix *Vec;    //�����һ�������ת�����ɵ���������������ȡ��ϣ���������������
                    //����������
    Matrix *F;      //�����ڶ���
    Matrix *WF;     //F���Ȩֵ
    Matrix *WFGradient;//�ݶ�
    Matrix *AllWFGradient;//���ݶ�
    Matrix *Out;    //���һ�㣬�������
    Matrix *WOut;   //Ȩֵ
    Matrix *WOutGradient;//�ݶ�
    Matrix *AllWOutGradient;//���ݶ�

};

#endif // CNN_NET_H_INCLUDED
