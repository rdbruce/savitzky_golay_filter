#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <thread>
#include <deque>
#include <future>
#include <boost/core/demangle.hpp>

using namespace std;

namespace py = pybind11;

class SavGol
{
public:
    // Python input
    Eigen::VectorXd data_in;
    int window_size;
    int smoothing_degree;
    int num_elements;
    int return_degree;
    bool threaded;

    // C++ vars
    Eigen::ArrayXi xWin;
    Eigen::MatrixXd conCo;

    SavGol(const py::kwargs& kwargs)
    {
        data_in = py::cast<Eigen::VectorXd>(kwargs["data_in"]);
        window_size = py::cast<int>(kwargs["window_size"]);
        smoothing_degree = py::cast<int>(kwargs["smoothing_degree"]);
        num_elements = py::cast<int>(kwargs["num_elements"]);
        return_degree = py::cast<int>(kwargs["return_degree"]);
        threaded = py::cast<bool>(kwargs["threaded"]);

        // Make our normalized X window (if win_size = 5 then xWin = -2, -1, 0, 1, 2)
        Eigen::ArrayXi xPreWin(window_size);
        for(int i = 0; i < window_size; i++)
        {
            xPreWin(i) = i + (1 - window_size)/2;
        }
        xWin = xPreWin;

        // Make corelation coefficients
        Eigen::MatrixXd van(window_size, smoothing_degree + 1);
        for(int j = 0; j <= smoothing_degree; j++)
        {
            van(Eigen::all, j) = xWin.pow(j).cast<double>();
        }
        conCo = (van.transpose()*van).inverse() * van.transpose();
    }

    deque<double> makePolynomial(int windowIndex)
    {
        Eigen::VectorXd yWin(window_size);
        yWin = data_in(Eigen::seq(windowIndex - window_size/2, windowIndex + window_size/2));

        Eigen::ArrayXd tmp = conCo*yWin;
        deque<double> vecA;
        for(int i = 0; i <= smoothing_degree; i++)
        {
            vecA.push_back(tmp(i));
        }
        return vecA;
    }

    void makeDerivative(deque<double>& coeffs)
    {
        int i = 0;
        while(i < return_degree)
        {
            coeffs.pop_front();
            for(int j = 0; j < coeffs.size(); j++)
            {
                coeffs[j] *= (j + 1);
            }
            i++;
        }
    }

    double eval(const deque<double>& coeffs, const int& z)
    {
        double bigy = 0;
        for(int i = 0; i < coeffs.size(); i++)
        {
            bigy += coeffs[i]*pow(z, i);
        }
        return bigy;
    }

    // Start processing
    void beginProcess(Eigen::VectorXd& returnData)
    {
        deque<double> coeffs = makePolynomial(window_size/2);
        makeDerivative(coeffs);

        for(int i = 0; i < window_size/2; i++)
        {
            cout << "******" << i << "******" << endl;
            returnData(i) = eval(coeffs, xWin(i));
        }
        cout << "end beginning" << endl;
    }

    void middleProcess(Eigen::VectorXd& returnData, int start, int end)
    {
        for(int i = start; i < end; i++)
        {
            deque<double> coeffs = makePolynomial(i);
            makeDerivative(coeffs);
            cout << "******" << i << "******" << endl;
            returnData(i) = eval(coeffs, xWin(window_size/2));
        }
        cout << "end middle" << endl;
    }

    void matrixMiddle(Eigen::VectorXd& returnData)
    {
        // for(int i = window_size/2; i < num_elements - window_size/2; i++)
        // {
        //     cout << "******" << i << "******" << endl;
        //     returnData(i) = tgamma(return_degree - 1) * (conCo.row(return_degree)).dot(data_in(Eigen::seq(i - window_size/2, i + window_size/2)));
        // }

        int output_size = num_elements - window_size + 1; //rows

        typedef Eigen::Triplet<double> T;
        std::vector<T> tripletList;
        tripletList.reserve(window_size * output_size);

        for(int i = 0; i < output_size; i++)
        {
            for(int j = 0; j < window_size; j++)
            {
                tripletList.push_back(T(i, j + i, conCo(return_degree, j)));
            }
        }
        Eigen::SparseMatrix<double, Eigen::RowMajor> mat(output_size, num_elements);
        mat.setFromTriplets(tripletList.begin(), tripletList.end());

        returnData(Eigen::seq(window_size/2, num_elements - window_size/2)) = mat * data_in;
        cout << "end matrix middle" << endl;
    }

    void endProcess(Eigen::VectorXd& returnData)
    {
        deque<double> coeffs = makePolynomial(num_elements - window_size/2 - 1);
        makeDerivative(coeffs);
        int j = window_size/2 + 1;
        for(int i = num_elements - window_size/2; i < num_elements; i++)
        {
            cout << "******" << i << "******" << endl;
            returnData(i) = eval(coeffs, xWin(j));
            j++;
        }
        cout << "end end" << endl;
    }

    Eigen::VectorXd filter()
    {
        Eigen::VectorXd returnData = Eigen::VectorXd::Zero(num_elements);
        if(!threaded)
        {
            beginProcess(returnData);
            // middleProcess(returnData, window_size/2, num_elements - window_size/2);
            matrixMiddle(returnData);
            endProcess(returnData);
        }
        else if(threaded)
        {
            int processor_count = std::thread::hardware_concurrency();
            async([&](){beginProcess(returnData);});
            processor_count--;
            async([&](){endProcess(returnData);});
            processor_count--;

            int middleElements = num_elements - window_size + 1;
            int middleElementBlock = middleElements/processor_count;
            cout << middleElements << " elements split into " << middleElementBlock << " element blocks." << endl;
            int start = window_size/2;
            int end = start + middleElementBlock;
            for(int i = 0; i < processor_count; i++)
            {
                if(i == processor_count - 1)
                {
                    end = num_elements - window_size/2;
                }
                async([&](){middleProcess(returnData, start, end);});
                start = start + middleElementBlock;
                end = start + middleElementBlock;
            }
        }
        return std::move(returnData);
    }
};

PYBIND11_MODULE(savitzky_golay, m) {
    m.doc() = "Please Work";

    py::class_<SavGol>(m, "SavGol")
        .def(py::init<const py::kwargs&>())
        .def("filter", &SavGol::filter);
}
