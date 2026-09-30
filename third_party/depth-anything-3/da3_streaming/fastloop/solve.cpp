// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

Eigen::VectorXd solve(const SpMat &A, const Eigen::VectorXd &b, int freen){

  if (freen < 0){
    const Eigen::SimplicialCholesky<SpMat> chol(A);
    return chol.solve(b); // n x 1
  }

  const SpMat A_sub = A.topLeftCorner(freen, freen);
  const Eigen::VectorXd b_sub = b.topRows(freen);
  const Eigen::VectorXd delta = solve(A_sub, b_sub, -7);

  Eigen::VectorXd delta2(b.rows());
  delta2.setZero();
  delta2.topRows(freen) = delta;

  return delta2;
}

std::vector<torch::Tensor> solve_system(torch::Tensor J_Ginv_i, torch::Tensor J_Ginv_j, torch::Tensor ii, torch::Tensor jj, torch::Tensor res, float ep, float lm, int freen)
{

  const torch::Device device = res.device();
  J_Ginv_i = J_Ginv_i.to(torch::kCPU);
  J_Ginv_j = J_Ginv_j.to(torch::kCPU);
  ii = ii.to(torch::kCPU);
  jj = jj.to(torch::kCPU);
  res = res.clone().to(torch::kCPU);

  const int r = res.size(0);
  const int n = std::max(ii.max().item<long>(), jj.max().item<long>()) + 1;

  res.resize_({r*7});
  float *res_ptr = res.data_ptr<float>();
  Eigen::Map<Eigen::VectorXf> v(res_ptr, r*7);

  SpMat J(r*7, n*7);
  std::vector<T> tripletList;
  tripletList.reserve(r*7*7*2);

  auto ii_acc = ii.accessor<long,1>();
  auto jj_acc = jj.accessor<long,1>();
  auto J_Ginv_i_acc = J_Ginv_i.accessor<float,3>();
  auto J_Ginv_j_acc = J_Ginv_j.accessor<float,3>();

  for (int x=0; x<r; x++){
    const int i = ii_acc[x];
    const int j = jj_acc[x];
    for (int k=0; k<7; k++){
      for (int l=0; l<7; l++){
        if (i == j)
          exit(1);
        const float val_i = J_Ginv_i_acc[x][k][l];
        tripletList.emplace_back(x*7 + k, i*7 + l, val_i);
        const float val_j = J_Ginv_j_acc[x][k][l];
        tripletList.emplace_back(x*7 + k, j*7 + l, val_j);
      }
    }
  }

  J.setFromTriplets(tripletList.begin(), tripletList.end());
  const SpMat Jt = J.transpose();
  Eigen::VectorXd b = -(Jt * v.cast<double>());
  SpMat A = Jt * J;

  A.diagonal() += (A.diagonal() * lm);
  A.diagonal().array() += ep;
  Eigen::VectorXf delta = solve(A, b, freen*7).cast<float>();

  torch::Tensor delta_tensor = torch::from_blob(delta.data(), {n*7}).clone().to(device);
  delta_tensor.resize_({n, 7});
  return {delta_tensor};

  Eigen::Matrix<float, -1, -1, Eigen::RowMajor> dense_J(J.cast<float>());
  torch::Tensor dense_J_tensor = torch::from_blob(dense_J.data(), {r*7, n*7}).clone().to(device);
  dense_J_tensor.resize_({r, 7, n, 7});

  return {delta_tensor, dense_J_tensor};

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("solve_system", &solve_system, "temporal neighboor indicies");
}
