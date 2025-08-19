#include "./Optimizer.h"
#include <functional>


Optim::SGD::SGD(double learning_rate, std::vector<std::reference_wrapper<Tensor>> save_tensors_)
: learning_rate(learning_rate), save_tensors_(save_tensors_) {}

void Optim::SGD::zero_grads() {
    for (Tensor& tensor : save_tensors_) {
        (*tensor.grad) = std::vector<double>(tensor.data->size(), 0.0);
    }
}

void Optim::SGD::apply_grads() {
    for (Tensor& tensor : save_tensors_) {
        for (size_t i = 0; i < tensor.data->size(); i++) {
            (*tensor.data)[i] = (*tensor.data)[i] - (learning_rate * (*tensor.grad)[i]);
        }
    }
}