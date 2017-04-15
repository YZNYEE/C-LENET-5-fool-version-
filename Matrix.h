#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED

/*矩阵类，BP与CNN都是应用的这个矩阵类完成的运算*/
class Matrix
{
public:
    Matrix(int r,int c);
    ~Matrix();
    //创建全1矩阵，r，c是维度
    static Matrix * CreateOnes(int r,int c);
    //创建0矩阵
    static Matrix * CreateZeros(int r,int c);
    //创建0-1随机矩阵
    static Matrix * CreateRand(int,int);
    //从double **类型转换为矩阵,这个函数基本多余
    static Matrix * CreateFromDouble(double **a,int r,int c);
    //矩阵相加
    static Matrix * Add(Matrix *,Matrix &,Matrix &);
    //矩阵与实数相加，即每个元数加实数
    static Matrix * Add(Matrix *,double ,Matrix &);
    //多个矩阵相加返回一个矩阵
    static Matrix * AddMulti(Matrix *,int ,Matrix **);
    //矩阵相减
    static Matrix * Minus(Matrix *,Matrix &,Matrix &);
    //实数减矩阵
    static Matrix * Minus(Matrix *,double x,Matrix &);
    //矩阵转置
    static Matrix * Transpose(Matrix *,Matrix &);
    //矩阵点乘
    static Matrix * DotTime(Matrix *,Matrix &,Matrix &);
    //偏度点乘，以后解释，完全为了运算方便
    static Matrix * DotTime(Matrix *,Matrix &,Matrix &,int n);
    //叉乘
    static Matrix * Time(Matrix *,Matrix &,Matrix &);
    //实数与矩阵相乘
    static Matrix * Time(Matrix *,double,Matrix &);
    //实数数组与矩阵相乘，以后解释，即分别每个实数与相应列相乘
    static Matrix * DoubleTime(Matrix *,double *,Matrix &);
    //两个矩阵横向相连
    static Matrix * CombineX(Matrix *,Matrix &,Matrix &);
    //竖向相连
    static Matrix * CombineY(Matrix *,Matrix &,Matrix &);
    //矩阵扩展，在计算梯度，s层返回到c层时用到
    static Matrix * CombineAll(Matrix *,int,Matrix &);
    //多个矩阵横向相连
    static Matrix * CombineXMulti(Matrix *,int,Matrix &);
    //矩阵sigmoid函数
    static Matrix * Sigm(Matrix *,Matrix &);
    //矩阵Log函数，在求误差时用到
    static Matrix * Log(Matrix *,Matrix &);
    //矩阵列项求和，返回一个1*n的矩阵
    static Matrix * Sum(Matrix *,Matrix &);
    //矩阵所有元数求和
    static Matrix * SumAll(Matrix *,Matrix &);
    //卷积运算
    static Matrix * Conv(Matrix *,Matrix &,Matrix &);
    //逆卷积运算 在反向传到时运算
    static Matrix * ObConv(Matrix *,Matrix &,Matrix &);
    //pooling运算
    static Matrix * Pooling(Matrix *,int,Matrix &);
    //多个矩阵转化为向量，在最后一层到BP网是用到
    static Matrix * TMToVector(Matrix *,int,Matrix **);
    //向量转化为多个矩阵，在反向传到时用到
    static Matrix ** VectorToMatrix(Matrix **,int,int,int,Matrix &);
    //获取n*1矩阵中最大的元数
    int GetMaxIndexVec();
    //矩阵置零
    void SetZeros();

    //以下是无关紧要的函数
    int ReR() const
    {
        return r;
    };
    int ReC() const
    {
        return c;
    };
    double *GetR(int x)
    {
        return matrix[x];
    }
    double GetNum(int n,int m) const
    {
        return matrix[n][m];
    }
    void Show();
//private:
    //矩阵类的成员
    double ** matrix;
    int r;
    int c;
};


#endif // MATRIX_H_INCLUDED
