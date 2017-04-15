#include <iostream>
#include "Matrix.h"
#include "stdlib.h"
#include "CNN_net.h"
#include "stdio.h"
#include "BP.h"
using namespace std;


int main()
{
      // CNN *net=new CNN();
      CNN *net=CNN::GetNetFromFile();
      //Matrix **x;
      //Matrix **y;
      //CNN::GetExample(&x,&y,60000);
      //(*net).Train(x,2,60000,50,0.05,y);
      Matrix **x1;
      Matrix **y1;
      CNN::GetExample1(&x1,&y1,10000);
      (*net).Test(x1,y1,10000);
      (*net).WriteNetInFile();
      //CNN *net1;
      //net1=CNN::GetNetFromFile();
}
