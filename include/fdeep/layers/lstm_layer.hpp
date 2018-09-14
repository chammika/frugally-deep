// Copyright 2016, Tobias Hermann.
// https://github.com/Dobiasd/frugally-deep
// Distributed under the MIT License.
// (See accompanying LICENSE file or at
//  https://opensource.org/licenses/MIT)

#pragma once

#include "fdeep/layers/layer.hpp"

#include <fplus/fplus.hpp>

#include <string>
#include <functional>

namespace fdeep
{
namespace internal
{

class lstm_layer : public layer
{
  public:
    explicit lstm_layer(const std::string &name,
                        std::size_t n_units,
                        std::size_t n_features,
                        const std::string &activation,
                        const std::string &recurrent_activation,
                        bool use_bias,
                        bool return_sequences,
                        const float_vec& weights, // const RowMajorMatrixXf &W,
                        const float_vec& recurrent_weights, // const RowMajorMatrixXf &U,
                        const float_vec& bias // const RowMajorMatrixXf &bias
                        )
        : layer(name),
          n_units_(n_units),
          activation_(activation),
          recurrent_activation_(recurrent_activation),
          act_func(get_activation_func(activation)),
          act_func_recurrent(get_activation_func(recurrent_activation_)),
          use_bias_(use_bias),
          return_sequences_(return_sequences),
          W_(eigen_mat_from_values(n_features, 4*n_units, weights)),  // weights shape (n_features, 4*units)
          U_(eigen_mat_from_values(n_units, 4*n_units, recurrent_weights)), // recurrent_weights shape (units, 4*units)
          bias_(eigen_mat_from_values(1, 4*n_units, bias))
    {
        // DEBUG
        std::cout << "weights:\n" << W_ << "\nrecurrent wieghts:\n" << U_ << "\nbais:\n" << bias_ << std::endl;
    }

  protected:
    tensor3s apply_impl(const tensor3s &inputs) const override final
    {
        assertion(inputs.front().shape().width_ == 1, "width dimension must be 1");
        assertion(inputs.front().shape().depth_ == 1, "depth dimension must be 1");

        return {lstm_impl(inputs)};
    }

  private:
    static float_type relu_activation(float_type x)
    {
        return std::max<float_type>(x, 0);
    }

    static float_type sigmoid_activation(float_type x)
    {
        return 1 / (1 + std::exp(-x));
    }

    static float_type hard_sigmoid_activation(float_type x)
    {
        return static_cast<float_type>(std::min(1.0, std::max(0.0, (0.2 * x) + 0.5)));
    }

    static float_type tanh_activation(float_type x)
    {
        return std::tanh(x);
    }

    std::function<float_type(float_type)> get_activation_func(const std::string &activation_func_name) const
    {
        if (activation_func_name == "relu")
            return relu_activation;
        else if (activation_func_name == "sigmoid")
            return sigmoid_activation;
        else if (activation_func_name == "hard_sigmoid")
            return hard_sigmoid_activation;
        else if (activation_func_name == "tanh")
            return tanh_activation;

        raise_error("activation function '" + activation_func_name + "' not yet implemented");
        return {};
    }

    tensor3s lstm_impl(const tensor3s &input) const
    {
        // initialize cell output states h, and cell memory states c for t-1 with zeros
        RowMajorMatrixXf h_tm1(1, n_units_);
        RowMajorMatrixXf c_tm1(1, n_units_);
        h_tm1.setZero();
        c_tm1.setZero();

        const std::size_t n_timesteps = input.size();
        const std::size_t n_features = input.front().shape().height_; // TODO - use depth (height =1 , width = 1)

        // write input to eigen matrix
        RowMajorMatrixXf in(n_timesteps, n_features);

        for (std::size_t a_t = 0; a_t < n_timesteps; ++a_t)
            for (std::size_t a_f = 0; a_f < n_features; ++a_f)
                in(EigenIndex(a_t), EigenIndex(a_f)) = input[a_t].get(0, a_f, 0); // TODO - get appropriately

        RowMajorMatrixXf X(n_timesteps, n_units_ * 4);
        X = in * W_;

        if (use_bias_)
        {
            // TODO - bias_ can be saved directly in Vector_Xf form by constructor
            // define eigen vector type to be able to use broadcasting
            typedef Eigen::Matrix<float_type, 1, Eigen::Dynamic> Vector_Xf;
            Vector_Xf b(1, n_units_ * 4);
            b = bias_;

            X.rowwise() += b;
        }

        // computing LSTM output
        const EigenIndex n = EigenIndex(n_units_);
        tensor3s result;
        for (EigenIndex k = 0; k < EigenIndex(n_timesteps); ++k)
        {
            const RowMajorMatrixXf ifco = h_tm1 * U_;

            // Use of Matrix.block(): Block of size (p,q), starting at (i,j) matrix.block(i,j,p,q);  matrix.block<p,q>(i,j);
            const RowMajorMatrixXf i = (X.block(k, 0, 1, n) + ifco.block(0, 0, 1, n)).unaryExpr(act_func_recurrent);
            const RowMajorMatrixXf f = (X.block(k, n, 1, n) + ifco.block(0, n, 1, n)).unaryExpr(act_func_recurrent);
            const RowMajorMatrixXf c_pre = (X.block(k, n * 2, 1, n) + ifco.block(0, n * 2, 1, n)).unaryExpr(act_func);
            const RowMajorMatrixXf o = (X.block(k, n * 3, 1, n) + ifco.block(0, n * 3, 1, n)).unaryExpr(act_func_recurrent);

            c_tm1 = f.cwiseProduct(c_tm1) + i.cwiseProduct(c_pre);
            h_tm1 = o.cwiseProduct(c_tm1.unaryExpr(act_func));

            // save every h sequence or just last
            if (return_sequences_)
                result.push_back(tensor3(shape3(1, n_units_, 1), eigen_mat_to_values(h_tm1)));
            else if (k == EigenIndex(n_timesteps) - 1)
                result = {tensor3(shape3(1, n_units_, 1), eigen_mat_to_values(h_tm1))};
        }

        return result;
    }

    const std::size_t n_units_;
    const std::string activation_;
    const std::string recurrent_activation_;
    const std::function<float_type(float_type)>& act_func;
    const std::function<float_type(float_type)>& act_func_recurrent;
    const bool use_bias_;
    const bool return_sequences_;
    const RowMajorMatrixXf W_;
    const RowMajorMatrixXf U_;
    const RowMajorMatrixXf bias_;
};

} // namespace internal
} // namespace fdeep
