import csv
import numpy as np
from biosppy.signals import ecg

def get_file_name(index):
  if index < 10:
    file = "0000"
  elif index > 9 and index < 100:
    file = "000"
  elif index > 99 and index < 1000:
    file = "00"
  else:
    file = "0"
  file = file + str(index)
  return file

def calculate_features(signal):
  sampling_rate = 400
  ts, signal, rpeaks, templates_ts, templates, heart_rate_ts, heart_rates = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)
  rpeaks = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)[0]
  heartbeat_templates, heartbeat_rpeaks = ecg.extract_heartbeats(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate)

  rpeaks_diff = np.diff(rpeaks)
  rpeaks_diff_diff = np.diff(rpeaks_diff)
  heartbeat_rpeaks_diff = np.diff(heartbeat_rpeaks)
  feature = ""

  mean_amplitude = np.mean(signal)
  std_amplitude = np.std(signal)
  max_amplitude = np.amax(signal)
  min_amplitude = np.amin(signal)
  median_amplitude = np.median(signal)
  feature = feature + str(mean_amplitude) + "," + str(std_amplitude) + "," + str(max_amplitude) + "," + str(min_amplitude) + "," + str(median_amplitude)

  mean_rpeaks_diff = np.mean(rpeaks_diff)
  std_rpeaks_diff = np.std(rpeaks_diff)
  median_rpeaks_diff = np.median(rpeaks_diff)
  feature = feature + "," + str(mean_rpeaks_diff) + "," + str(std_rpeaks_diff) + "," + str(median_rpeaks_diff)

  mean_rpeaks_diff_diff = np.mean(rpeaks_diff_diff)
  std_rpeaks_diff_diff = np.std(rpeaks_diff_diff)
  median_rpeaks_diff_diff = np.median(rpeaks_diff_diff)
  feature = feature + "," + str(mean_rpeaks_diff_diff) + "," + str(std_rpeaks_diff_diff) + "," + str(median_rpeaks_diff_diff)

  heartbeat_length = len(heartbeat_templates)
  mean_heart_rate = np.mean(heart_rates) if len(heart_rates) > 0 else 0.0
  std_heart_rate = np.std(heart_rates) if len(heart_rates) > 0 else 0.0
  median_heart_rate = np.median(heart_rates) if len(heart_rates) > 0 else 0.0
  feature = feature + "," + str(heartbeat_length) + "," + str(mean_heart_rate) + "," + str(std_heart_rate) + "," + str(median_heart_rate)

  mean_heartbeat_rpeaks_diff = np.mean(heartbeat_rpeaks_diff)
  std_heartbeat_rpeaks_diff = np.std(heartbeat_rpeaks_diff)
  median_heartbeat_rpeaks_diff = np.median(heartbeat_rpeaks_diff)
  feature = feature + "," + str(mean_heartbeat_rpeaks_diff) + "," + str(std_heartbeat_rpeaks_diff) + "," + str(median_heartbeat_rpeaks_diff)
  return feature

def main():
  N = 7000
  T = 1528
  writer = open('features.csv', 'w')
  for index in range(N):
    file = get_file_name(index+1)
    signal = np.loadtxt('training/A' + str(file) + '.txt')
    feature = calculate_features(signal)
    writer.write(feature + '\n')
  writer.close()
  writer = open('test_set.csv', 'w')
  for index in range(T):
    file = get_file_name(index+1)
    signal = np.loadtxt('testing/A' + str(file) + '.txt')
    feature = calculate_features(signal)
    writer.write(feature + '\n')
  writer.close()

if __name__ == "__main__":
  main()