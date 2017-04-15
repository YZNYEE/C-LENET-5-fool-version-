#include"BP.h"
#include"Matrix.h"
#include<iostream>
#include "stdio.h"
using namespace std;

BP::BP(int n):Size(n)
{
    F=Matrix::CreateZeros(n*3,1);
    WF=Matrix::CreateRand(n*3,n+1);
    WFGradient=Matrix::CreateZeros(n*3,n+1);
    AllWFGradient=Matrix::CreateZeros(n*3,n+1);
    Out=Matrix::CreateZeros(10,1);
    WOut=Matrix::CreateRand(10,n*3+1);
    WOutGradient=Matrix::CreateZeros(10,n*3+1);
    AllWOutGradient=Matrix::CreateZeros(10,n*3+1);
}

void BP::FeedForward(Matrix &x)
{
    Matrix *One=Matrix::CreateOnes(1,1);
    Matrix *OneAddx=nullptr;
    OneAddx=Matrix::CombineY(OneAddx,*One,x);
    F=Matrix::Time(F,*WF,*OneAddx);
    F=Matrix::Sigm(F,*F);
    Matrix *FAddOne=nullptr;
    FAddOne=Matrix::CombineY(FAddOne,*One,*F);
    Out=Matrix::Time(Out,*WOut,*FAddOne);
    Out=Matrix::Sigm(Out,*Out);
}

void BP::ComputeGradient(Matrix &x,Matrix &y)
{
    Matrix *Mns=nullptr;
    Mns=Matrix::Minus(Mns,*Out,y);
    Matrix *MnsMulti=nullptr;
    MnsMulti=Matrix::CombineXMulti(MnsMulti,F->r,*Mns);
    Matrix *Ft=nullptr;
    Ft=Matrix::Transpose(Ft,*F);
    Matrix *ThetaOut=nullptr;
    ThetaOut=Matrix::DoubleTime(ThetaOut,Ft->matrix[0],*MnsMulti);
    WOutGradient=Matrix::CombineX(WOutGradient,*Mns,*ThetaOut);

    Matrix *ThetaF=nullptr;
    ThetaF=Matrix::DotTime(ThetaF,*WOut,*MnsMulti,1);
    Matrix *SumThetaF=nullptr;
    SumThetaF=Matrix::Sum(SumThetaF,*ThetaF);
    Matrix *SumThetaFt=nullptr;
    SumThetaFt=Matrix::Transpose(SumThetaFt,*SumThetaF);
    Matrix *OneMnsF=nullptr;
    OneMnsF=Matrix::Minus(OneMnsF,1,*F);
    SumThetaFt=Matrix::DotTime(SumThetaFt,*SumThetaFt,*F);
    SumThetaFt=Matrix::DotTime(SumThetaFt,*SumThetaFt,*OneMnsF);
    OneMnsF->~Matrix();
    Matrix *SumThetaFtMulti=nullptr;
    SumThetaFtMulti=Matrix::CombineXMulti(SumThetaFtMulti,x.r,*SumThetaFt);
    Matrix *xt=nullptr;
    xt=Matrix::Transpose(xt,x);
    Matrix *ThetaWF=nullptr;
    ThetaWF=Matrix::DoubleTime(ThetaWF,xt->matrix[0],*SumThetaFtMulti);
    WFGradient=Matrix::CombineX(WFGradient,*SumThetaFt,*ThetaWF);
    Mns->~Matrix();
    MnsMulti->~Matrix();
    Ft->~Matrix();
    ThetaOut->~Matrix();
    ThetaF->~Matrix();
    SumThetaF->~Matrix();
    SumThetaFt->~Matrix();
    SumThetaFtMulti->~Matrix();
    xt->~Matrix();
    ThetaWF->~Matrix();
}

void BP::DGradient(int m,double Efficiency)
{
    AllWFGradient=Matrix::Time(AllWFGradient,Efficiency/m,*AllWFGradient);
    WF=Matrix::Minus(WF,*WF,*AllWFGradient);
    AllWOutGradient=Matrix::Time(AllWOutGradient,Efficiency/m,*AllWOutGradient);
    WOut=Matrix::Minus(WOut,*WOut,*AllWOutGradient);
}

void BP::SetZeros()
{
    AllWFGradient->SetZeros();
    AllWOutGradient->SetZeros();
}

void BP::Train(Matrix **x,int num,int n,double Efficiency,Matrix **y)
{
    for(int i=0;i<num;i++)
    {
        SetZeros();
        double Error=0;
        for(int j=0;j<n;j++)
        {
            FeedForward(*x[j]);
            Out->Show();
            getchar();
            getchar();
            ComputeGradient(*x[j],*y[j]);
            ApplyGradient();
            Error+=GetError(*y[j]);
        }
        DGradient(n,Efficiency);
        Error/=n;
        cout<<Error<<endl;
    }

}

double BP::GetError(Matrix &Y)
{
    Matrix *OneY=nullptr;
    OneY=Matrix::Minus(OneY,1,Y);
    Matrix *OneOut=nullptr;
    OneOut=Matrix::Minus(OneOut,1,*Out);
    Matrix *LogOneOut=nullptr;
    LogOneOut=Matrix::Log(LogOneOut,*OneOut);
    Matrix *LogOut=nullptr;
    LogOut=Matrix::Log(LogOut,*Out);
    Matrix *TimeOutY=nullptr;
    TimeOutY=Matrix::DotTime(TimeOutY,Y,*LogOut);
    Matrix *TimeOneOutY=nullptr;
    TimeOneOutY=Matrix::DotTime(TimeOneOutY,*OneY,*LogOneOut);
    Matrix *SumEr=nullptr;
    SumEr=Matrix::Add(SumEr,*TimeOutY,*TimeOneOutY);
    Matrix *SumErAll=nullptr;
    SumErAll=Matrix::SumAll(SumErAll,*SumEr);
    double ans=SumErAll->matrix[0][0]*(-1.0);
    return ans;
}

void BP::Test(Matrix **x,Matrix **y,int n)
{
    double ans=0;
    for(int i=0;i<n;i++)
    {
        FeedForward(*x[i]);
        if(Out->GetMaxIndexVec()==y[i]->GetMaxIndexVec())
            ans++;
    }
    cout<<ans/n<<endl;
}

void BP::ApplyGradient()
{
    AllWFGradient=Matrix::Add(AllWFGradient,*AllWFGradient,*WFGradient);
    AllWOutGradient=Matrix::Add(AllWOutGradient,*AllWOutGradient,*WOutGradient);
}
