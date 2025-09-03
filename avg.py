from Main import EnhancedLeakDetector
import matplotlib.pyplot as plt



class avg(EnhancedLeakDetector):

    def starter(self, path, bpath):
        self.time, self.data = self.load_and_preprocess(path)
        mask = self.time <= 1.0
        cropped_data = self.data[mask]
        cropped_time = self.time[mask]
        avg_value = cropped_data.mean()
        print("Average of signal in first 1s:", avg_value)
     
        

        plt.figure(figsize=(10, 5))
        plt.plot(cropped_time, cropped_data, label='Signal (first 1s)')
        plt.axhline(avg_value, color='r', linestyle='--', label=f'Avg Signal: {avg_value:.2f}')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Signal and Baseline (First 1s) with Averages')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    file_path = 'C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\Before avg\\Baseline.csv'
    before_avg_data = 'C:\\Users\\abdul\\Desktop\\Hydrophone_Project\\Response_Folder\\analogbaseline.csv'
    x = avg()
    x.starter(file_path, before_avg_data)
