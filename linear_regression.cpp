#include <iostream>
#include <vector>
#include <cmath>
#include <utility> 

class DNN
{
    public:
    std::pair<double, double> output;
    double cost = 0;

    DNN(const std::vector<int> &x_, const std::vector<int> &y_, const int &epoch_, const float &lr_)
    {
        x = x_;
        y = y_;
        epoch = epoch_;
        lr = lr_;
        output = DNN::learning_function(x,y);
    }

    private:
    std::vector<int> x;
    std::vector<int> y;

    double w;
    double b;

    double sigma_w_gradient;
    double sigma_b_gradient;

    int count;

    int epoch;
    float lr;

    std::pair<double, double> learning_function(const std::vector<int> &x, const std::vector<int> &y)
    {
        for(int i = 0; i < epoch; i++)
        {
            sigma_w_gradient += (2 * std::pow(x[count],2) * w) + (2 * x[count] * b) - (2 * x[count] * y[count]);
            sigma_b_gradient += (2 * x[count] * w) - (2 * y[count]) + (2 * b);

            if(count == (x.size() - 1))
            {
                sigma_w_gradient /= x.size();
                sigma_b_gradient /= x.size();

                w -= (lr * sigma_w_gradient);
                b -= (lr * sigma_b_gradient);

                sigma_w_gradient = 0;
                sigma_b_gradient = 0;
            }

            count++;
            if(count == x.size())
            {
                count = 0;
            }

            // if((i % 500) == 0)
            // {
            //     std::cout << "w : " << w << "           " << "b : " << b << "           "  << "cost :" << (w * x[count] + b) - y[count] <<std::endl;
            // }
            cost = (w * x[count] + b) - y[count];
        }

        return std::make_pair(w,b);
    }

};

int main()
{
    int epoch = 5000;
    float lr = 0.01;

    DNN data_obj({1,2,3},{2,4,6},epoch,lr);

    std::cout << "Equation : y = " << data_obj.output.first << "x + " << data_obj.output.second <<std::endl;
    std::cout << "Cost : " << data_obj.cost << std::endl;

    return 0;
}
