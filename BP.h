#ifndef BP_H_INCLUDED
#define BP_H_INCLUDED
#include"Matrix.h"

//bp网只设了两层，函数同CNN

class BP
{
public:
BP(int);
void FeedForward(Matrix &);
void ComputeGradient(Matrix &,Matrix &);
void DGradient(int,double);
void Train(Matrix **x,int num,int n,double Efficiency,Matrix **y);
void GetExample(Matrix ***,Matrix ***,int n);
void GetExample1();
double GetError(Matrix &Y);
void SetZeros();
void Test(Matrix **x,Matrix **y,int n);
void ApplyGradient();

int Size;

Matrix *F;
Matrix *WF;
Matrix *WFGradient;
Matrix *AllWFGradient;

Matrix *Out;
Matrix *WOut;
Matrix *WOutGradient;
Matrix *AllWOutGradient;
};


#endif // BP_H_INCLUDED
