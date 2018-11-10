// Copyright (C) 2010, 2011, 2012, 2013, 2014 Steffen Rendle
// Contact:   srendle@libfm.org, http://www.libfm.org/
//
// This file is part of libFM.
//
// libFM is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libFM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libFM.  If not, see <http://www.gnu.org/licenses/>.
//
//
// fm_learn_sgd.h: Stochastic Gradient Descent based learning for
// classification and regression
//
// Based on the publication(s):
// - Steffen Rendle (2010): Factorization Machines, in Proceedings of the 10th
//   IEEE International Conference on Data Mining (ICDM 2010), Sydney,
//   Australia.

#ifndef FM_LEARN_SGD_ELEMENT_H_
#define FM_LEARN_SGD_ELEMENT_H_

#include "fm_learn_sgd.h"

class fm_learn_sgd_element: public fm_learn_sgd {
 public:
  virtual void init();
  virtual void learn(Data& train, Data& test);
};

// Implementation
void fm_learn_sgd_element::init() {
  fm_learn_sgd::init();

  if (log != NULL) {
    log->addField("rmse_train", std::numeric_limits<double>::quiet_NaN());
  }
}

void fm_learn_sgd_element::learn(Data& train, Data& test) {
	// wk_debug
	//std::cout << "*** fm_learn_sgd_element::learn() ***" << std::endl;
	//std::cout << "*** num_iter: " << num_iter << " ***" << std::endl;
	//std::cout << "*** num_cases: " << train.num_cases << " ***" << std::endl;

  fm_learn_sgd::learn(train, test);

#ifdef ENABLE_MPI
	if (fm->my_rank == MPI_SERVER_NODE) {
		double server_time = 0;
		server_time -= MPI_Wtime();
		for (int i = 0; i < num_iter; i++) {
			for (int worker = 1; worker < fm->world_size; worker++) {
				fm->server_pull();
			}

			// 将ROOT节点的权值同步到所有worker节点
			fm->server_learn(learn_rate);
			fm->server_push();

			double rmse_train = evaluate(train);
			double rmse_test = evaluate(test);
			std::cout << "Server: #Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
		}

		server_time += MPI_Wtime();
		std::cout << "***** Total training time on server: " << server_time << " seconds" << std::endl;
		std::cout.flush();
		MPI_Finalize();

		return ;
	}
#endif

  std::cout << "SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS." << std::endl;
  // SGD
#ifdef ENABLE_MPI
    if (fm->world_size < 2)
	    throw "at least 2 process must be spawned!";
    int train_segment = (int)(train.num_cases / (fm->world_size-1));
    int start_index = train_segment * (fm->my_rank - 1);
    int end_index = start_index + train_segment;
    std::cout << "start index: " << start_index << ", end index: " << end_index << std::endl;

#endif
  for (int i = 0; i < num_iter; i++) {
    double iteration_time = getusertime();
#ifdef ENABLE_MPI
    //for (train.data->set(start_index); train.data->cur() < end_index; train.data->next()) {
	    //std::cout << "rand(): " << rand() << std::endl;
    int counter = 0;
    while (counter < train_segment) {
	    counter ++;
	    train.data->set(rand() % train.num_cases);
#else
    for (train.data->begin(); !train.data->end(); train.data->next()) {
#endif
      double p = fm->predict(train.data->getRow(), sum, sum_sqr);
      double mult = 0;
      if (task == 0) {
        p = std::min(max_target, p);
        p = std::max(min_target, p);
        mult = -(train.target(train.data->getRowIndex())-p);
      } else if (task == 1) {
        mult = -train.target(train.data->getRowIndex())*(1.0-1.0/(1.0+exp(-train.target(train.data->getRowIndex())*p)));
      }
      SGD(train.data->getRow(), mult, sum);
    }
    iteration_time = (getusertime() - iteration_time);
    //double rmse_train = evaluate(train);
    //double rmse_test = evaluate(test);
    //std::cout << "#Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
    if (log != NULL) {
      double rmse_train = evaluate(train);
      log->log("rmse_train", rmse_train);
      log->log("time_learn", iteration_time);
      log->newLine();
    }
#ifdef ENABLE_MPI
	if (fm->my_rank != MPI_SERVER_NODE) { // worker node
		fm->worker_push();	
		fm->worker_pull();

		/*
		double rmse_train = evaluate(train);
		double rmse_test = evaluate(test);
		std::cout << "Client: #Iter=" << std::setw(3) << i << "\tTrain=" << rmse_train << "\tTest=" << rmse_test << std::endl;
		*/
	}
#endif
  }
}

#endif /*FM_LEARN_SGD_ELEMENT_H_*/
