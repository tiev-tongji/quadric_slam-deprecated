#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include "vector"
#include <math.h>
#include <Eigen/Dense>


namespace  ds{
    class DPProcess{
       public:
        int totallClass;
        Eigen::VectorXd params;
        int sum ;
        double alpha; //para for  new class;
        DPProcess(double alpha):totallClass(0),alpha(alpha)
        {}

        double calProb(int classNum){
            if(classNum>totallClass) return 0.0;
            else if(classNum == totallClass) return alpha/(sum+alpha);
            else return params(classNum)/ (sum+alpha);
        }
        bool update(int classNum){
            if (classNum>=totallClass){
                return false;
            }
            else{
                params(classNum)++;
                sum++;
                return true;
            }
        }
        bool newClass(){
            totallClass++;
            params.resize(totallClass);
            params(totallClass-1) = 1.0;
            sum++;
        }
    };
};

    class  DirDS{
    public:
        int totallClass; // num of categories;
        Eigen::VectorXd params;
        DirDS(int totallClass):totallClass(totallClass){
            params.resize(totallClass);
        }
        bool update(int classNum){
            if(classNum>=totallClass) return false;
            else {
                params(classNum) ++;
                return true ;
            }
        }
        Eigen::VectorXd sampling(){
            return params;// Todo draw a sample from dir;
        }

    };

    class  CatDS
    {
    public:
        int totallClass;
        Eigen::VectorXd params;
        DirDS dir;
         CatDS(int totallClass):totallClass(totallClass) {
             params.resize(params);
             dir(totallClass);
         }
         bool update(int classNum){
             dir.update(classNum);
              params = dir.sampling();
         }
         double calProb(int classNum){
             if(classNum>=totallClass)return 0.0;
             else{
                 return params(classNum);
             }
         }
    };

#endif // DISTRIBUTION_HPP
