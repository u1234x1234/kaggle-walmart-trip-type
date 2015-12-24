#include <iostream>
#include <chrono>
#include <unordered_map>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace cv;

string root_path = "/home/dima/walmart/";
unordered_map<string, int> feature_to_id[6];
vector<int> feat_count(6);
unordered_map<string,int> week_day_mapping = {{"Monday", 0}, {"Tuesday", 1}, {"Wednesday", 2}, {"Thursday", 3}, {"Friday", 4}, {"Sunday", 5}, {"Saturday", 6}};
void read_test_stat()
{
	vector<string> header;
	string line;
	ifstream in2(root_path + "/prep_test.csv");
	getline(in2, line);
	boost::split(header, line, boost::is_any_of("_"));
	while(getline(in2, line))
	{
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of("_"));
		for (int i = 0; i < 6; i++)
		{
			if (feature_to_id[i].find(tokens[i]) == feature_to_id[i].end())
			{
				if (tokens[i] == "")
					cout << "nan: " << i << " " << feat_count[i] << endl;
				feature_to_id[i][tokens[i]] = feat_count[i]++;
			}
		}
	}
	for (int i = 0; i < 6; i++)
		cout << header[i] << " " << feature_to_id[i].size() << endl;
	cout << endl;
}

void read_train_stat()
{
	vector<string> header;
	string line;
	ifstream in2(root_path + "/prep_train.csv");
	getline(in2, line);
	boost::split(header, line, boost::is_any_of("_"));
	while(getline(in2, line))
	{
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of("_"));
		for (int i = 0; i < 6; i++)
		{
			if (feature_to_id[i].find(tokens[i+1]) == feature_to_id[i].end())
			{
				feature_to_id[i][tokens[i+1]] = feat_count[i]++;
			}
		}
	}
	for (int i = 0; i < 6; i++)
		cout << header[i+1] << " " << feature_to_id[i].size() << endl;
	cout << endl;
}

struct trip_info
{
	int weekday;
	int trip_type;
	vector<int> data[4];
};


map<int,trip_info> read_trips(string mode = "train")
{
	map<int,trip_info> trips;
	ifstream in(root_path + "/prep_" + mode + ".csv");
	int fl = (mode == "test");
	string line;
	getline(in, line);
	while(getline(in, line))
	{
		vector<string> tokens;
		boost::split(tokens, line, boost::is_any_of("_"));
		int trip_type = stoi(tokens[0]);
		int visit_number = stoi(tokens[1 - fl]);

//		int weekday = feature_to_id[1][tokens[2 - fl]];
		int weekday = week_day_mapping[tokens[2 - fl]];

		int dat[4];
		for (int i = 2; i < 6; i++)
		{
			if (feature_to_id[i].find(tokens[i + 1 - fl]) == feature_to_id[i].end())
				continue;
			if (i != 3)// scan_count is already in num format
				dat[i - 2] = feature_to_id[i][tokens[i + 1 - fl]];
			else
			{
				dat[i - 2] = stoi(tokens[i + 1 - fl]);
			}
		}

		auto &trip = trips[visit_number];
		trip.weekday = weekday;
		trip.trip_type = trip_type;
		for (int i = 0; i < 4; i++)
			trip.data[i].push_back(dat[i]);
	}
	return trips;
}

void gen_features(const map<int,trip_info> &trips, string mode = "train")
{
	ofstream out_indices(root_path + "/" + mode +".indices");
	ofstream out_indptr(root_path + "/" + mode + ".indptr");
	ofstream out_data(root_path + "/" + mode + ".data");
	ofstream out_y(root_path + "/" + mode + ".y");
	ofstream out(root_path + "/target_mapping");

	int size = 0;
	out_indptr.write(reinterpret_cast<char*>(&size), sizeof(size));
	for (auto &trip : trips)
	{
		int writed_cnt = 0;
		auto write = [&](int idx, float val) -> void
		{
			out_indices.write(reinterpret_cast<char*>(&idx), sizeof(idx));
			out_data.write(reinterpret_cast<char*>(&val), sizeof(val));
			writed_cnt++;
		};

		int ret_flag = 0;
		int nan_flag1 = 0;
		int nan_flag2 = 0;
		int nan_flag3 = 0;
		int scan_count = 0;
		int scan_count_sum = 0;
		int ret_count = 0;

		map<int, int> hist[4];
		map<int, int> hist_count;
		vector<int> cnts(3);
		for (size_t i = 0; i < trip.second.data[0].size(); i++)
		{
			if (trip.second.data[1][i] < 0) // return
			{
				ret_flag++;
				ret_count += (-trip.second.data[1][i]);
			}
			else
			{
				int q = trip.second.data[1][i];
				q = 1;

				hist[0][trip.second.data[0][i]] += q;
				hist[1][trip.second.data[2][i]] += q;
				hist[2][trip.second.data[3][i]] += q;
				hist_count[trip.second.data[1][i]]++;

				scan_count_sum += trip.second.data[1][i];
				scan_count++;
			}

			if (trip.second.data[0][i] == 563)
				nan_flag1++;
			if (trip.second.data[2][i] == 46)
				nan_flag1++;
			if (trip.second.data[3][i] == 418)
				nan_flag1++;
		}
		int num_unique = 0;
		int offset = 0;
		write(offset + trip.second.weekday, 1);
		offset += 7;
		for (int i = 0; i < 3; i++)
		{
			for (auto &it : hist[i])
			{
				write(offset + it.first, it.second / (float)scan_count);
				num_unique++;
			}
			if (i == 0)
				offset += 98148;
			if (i == 1)
				offset += 68;
			if (i == 2)
				offset += 5204;
		}
		for (int i = 0; i < 10; i++)
			write(offset + i, hist_count[i]);
		offset += 10;

//		for (auto &it1 : hist[1])
//			for (auto &it2 : hist[2])
//				write(offset + it1.first*5204 + it2.first, it1.second + it2.second);
//		offset += 68 * 5204;

//		write(offset++, num_unique);
		write(offset++, ret_count);
		write(offset++, trip.second.data[0].size());
		write(offset++, nan_flag1);
//		write(7 + 98148 + 68 + 5204 + 3, nan_flag2);
//		write(7 + 98148 + 68 + 5204 + 4, nan_flag3);
		write(offset++, scan_count);
		write(offset++, scan_count_sum);
		write(offset++, hist[0].size());
		write(offset++, hist[1].size());
		write(offset++, hist[2].size());


//		write(offset + 8, ret_count);

//		write(7 + 98148 + 68 + 5204 + 11, num_unique);


//		write(7 + 98148 + 68 + 5204 + 11, num_unique);
		//relative ratio

//		write(offset + 8, hist[0].size() / (float)num_unique);
//		write(offset + 9, hist[1].size() / (float)num_unique);
//		write(offset + 10, hist[2].size() / (float)num_unique);

//		write(7 + 98148 + 68 + 5204 + 11, trip.second.weekday >= 5);


		size += (writed_cnt);
		out_indptr.write(reinterpret_cast<char*>(&size), sizeof(size));

		int trip_type = trip.second.trip_type;
		out_y.write(reinterpret_cast<char*>(&trip_type), sizeof(trip.second.trip_type));
	}
}

int main()
{
	ifstream in(root_path + "/prep_train.csv");

	string line;
	vector<string> header;
	getline(in, line);
	boost::split(header, line, boost::is_any_of("_"));

	read_test_stat();
	//    read_train_stat();
	map<int,trip_info> train_trips = read_trips();
	map<int,trip_info> test_trips = read_trips("test");

	cout << "train_trip size: " << train_trips.size() << endl;
	cout << "test_trip size: " << test_trips.size() << endl;

	gen_features(train_trips);
	gen_features(test_trips, "test");


	return 0;
}


