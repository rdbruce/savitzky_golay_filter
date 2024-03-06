#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/framework/accumulator_set.hpp>

using namespace std;

namespace py = pybind11;

Eigen::VectorXi makeXWin(const int& win_size)
{
    Eigen::VectorXi xWin(win_size);
    for(int i = 0; i < win_size; i++)
    {
        xWin(i) = i + (1 - win_size)/2;
    }
    return xWin;
}

Eigen::MatrixXd makeConCo(const Eigen::VectorXi& xWin, const int& win_size, const int& deg)
{
    Eigen::MatrixXd van(win_size, deg + 1);
    for(int j = 0; j <= deg; j++)
    {
        for(int i = 0; i < win_size; i++)
        {
            van(i, j) = pow(xWin(i), j);
        }
    }

    Eigen::MatrixXd conCo;
    conCo = (van.transpose()*van).inverse() * van.transpose();
    return conCo;
}

Eigen::VectorXd makeYWin(const Eigen::VectorXd& data, const int& win_size, const int& start_loc)
{
    Eigen::VectorXd yWin(win_size);
    for(int i = 0; i < win_size; i++)
    {
        yWin(i) = data(start_loc + i);
    }
    return yWin;
}

Eigen::VectorXd makeVecA(const Eigen::MatrixXd& conCo, const Eigen::VectorXd& yWin, const int& deg)
{
    Eigen::VectorXd vecA(deg + 1);
    for(int i = 0; i <= deg; i++)
    {
        vecA(i) = (conCo.row(i)).dot(yWin);
    }
    return vecA;
}

Eigen::VectorXd savGolFilter(const py::kwargs& kwargs)
{
    auto data = py::cast<Eigen::VectorXd>(kwargs["data_in"]);
    const int window_size = py::cast<const int>(kwargs["window_size"]);
    const int smoothing_degree = py::cast<const int>(kwargs["smoothing_degree"]);
    const int num_elements = py::cast<const int>(kwargs["num_elements"]);
    const int return_degree = py::cast<const int>(kwargs["return_degree"]);
    
    Eigen::VectorXi xWin = makeXWin(window_size);
    Eigen::MatrixXd conCo = makeConCo(xWin, window_size, smoothing_degree);
    Eigen::VectorXd returnData = Eigen::VectorXd::Zero(num_elements);

    if(return_degree == 0)
    {
        // Assign edge values
        int start_loc = 0;
        Eigen::VectorXd yWin = makeYWin(data, window_size, start_loc);
        Eigen::VectorXd vecA = makeVecA(conCo, yWin, smoothing_degree);
        int returnElement = 0;
        while(returnElement < window_size/2)
        {
            for(int j = 0; j <= smoothing_degree; j++)
            {
                returnData(returnElement) += vecA(j)*pow(xWin(returnElement), j);
            }
            returnElement++;
        }
        start_loc++;

        // Assign middle values
        while(returnElement < (num_elements - window_size/2 - 1))
        {
            yWin = makeYWin(data, window_size, start_loc);
            vecA = makeVecA(conCo, yWin, smoothing_degree);
            returnData(returnElement) = vecA(0);

            returnElement++;
            start_loc++;
        }

        // Assign edge values
        int z = window_size/2;
        while(returnElement < num_elements)
        {
            for(int j = 0; j <= smoothing_degree; j++)
            {
                // cout << xWin << " element " << z << endl;
                // cout << xWin(z) << " ^ " << j << endl;
                // cout << vecA(j) << " * " << pow(xWin(z), j) << endl;
                // cout << returnData(returnElement) << " + " << vecA(j)*pow(xWin(z), j) << endl;
                returnData(returnElement) += vecA(j)*pow(xWin(z), j);
                // cout << returnData(returnElement) << endl; 
            }
            returnElement++;
            z++;
        }
    }
    else
    {
        int start_loc = 0;
        int returnElement = window_size/2;

        // Assign middle values
        while(returnElement < (num_elements - window_size/2 - 1))
        {
            Eigen::VectorXd yWin = makeYWin(data, window_size, start_loc);
            Eigen::VectorXd vecA = makeVecA(conCo, yWin, smoothing_degree);
            returnData(returnElement) = tgamma(return_degree + 1) * vecA(return_degree);

            returnElement++;
            start_loc++;
        }
    }

    return returnData;
}

PYBIND11_MODULE(savitzky_golay, m) {
    m.doc() = "savGolFilter(input_vector, window_size, smoothing_degree, return_degree)";

    m.def("savGolFilter", &savGolFilter, "This is a Savitzky Golay filter function.");
}
