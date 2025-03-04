import pandas as pd
import matplotlib.pyplot as plt

class Visualize:
    @staticmethod
    def main():

      data = pd.read_csv('data/visualization_data.csv')

      csci_adj = ['CSCI', 'CISS', 'DATA']
      csci_data = data[data['SUBJECT'].isin(csci_adj)]

      required_classes = ['141', '145', '301', '241', '247', '305', '311', '305', '330', '345', '347', '447', '367', '405', '346', '491', '492', '493', '421', '321', '342', '343', '372', '400', '401', '402', '404', '410', '412', '415', '424', '426', '430', '436', '440', '442', '450', '461', '462', '463', '467', '474', '477', '479', '480', '481', '496', '497', '471', '335', '375', '435']
      req_csci_data = csci_data[csci_data['COURSE_NUMBER'].isin(required_classes)]

      def plot_data(course_enrollment):
        course_enrollment_pivot = course_enrollment.pivot_table(index=['TITLE'], columns='Start_Date', values='ACTUAL_ENROLL', fill_value=0)

        # Sort by longest bars
        course_enrollment_pivot['Total_Enrollment'] = course_enrollment_pivot.sum(axis=1)
        course_enrollment_pivot = course_enrollment_pivot.sort_values(by='Total_Enrollment', ascending=False)
        course_enrollment_pivot = course_enrollment_pivot.drop(columns=['Total_Enrollment'])

        # Plot the diagram
        fig, ax = plt.subplots(figsize=(10, 30))
        course_enrollment_pivot.plot(kind='barh', stacked=True, ax=ax)

        ax.invert_yaxis()
        ax.set_xlabel('Number of Students')
        ax.set_ylabel('Course Title')
        ax.set_title('Number of Students Enrolled per Course')
        plt.legend(loc='upper right', title='Start Date')
        plt.show()


      course_enrollment = req_csci_data.groupby(['TITLE', 'Start_Date'])['ACTUAL_ENROLL'].sum().reset_index()

      #Get rid of classes that never have less than 10 students
      sum_enroll = course_enrollment.groupby('TITLE')['ACTUAL_ENROLL'].sum()
      valid_titles = sum_enroll[sum_enroll > 10].index
      course_enrollment = course_enrollment[course_enrollment['TITLE'].isin(valid_titles)]

      #Plot
      plot_data(course_enrollment)

      #FALL
      # req_csci_data_fall = req_csci_data[req_csci_data['TERM'].str.contains('Fall')]
      # course_enrollment_fall = req_csci_data_fall.groupby(['TITLE', 'Start_Date'])['ACTUAL_ENROLL'].sum().reset_index()
      # plot_data(course_enrollment_fall)

      # WINTER
      # req_csci_data_winter = req_csci_data[req_csci_data['TERM'].str.contains('Winter')]
      # course_enrollment_winter = req_csci_data_winter.groupby(['TITLE', 'Start_Date'])['ACTUAL_ENROLL'].sum().reset_index()
      # plot_data(course_enrollment_winter)

      # SPRING
      # req_csci_data_spring = req_csci_data[req_csci_data['TERM'].str.contains('Spring')]
      # course_enrollment_spring = req_csci_data_spring.groupby(['TITLE', 'Start_Date'])['ACTUAL_ENROLL'].sum().reset_index()
      # plot_data(course_enrollment_spring)

      #SUMMER
      # req_csci_data_summer = req_csci_data[req_csci_data['TERM'].str.contains('Summer')]
      # course_enrollment_summer = req_csci_data_summer.groupby(['TITLE', 'Start_Date'])['ACTUAL_ENROLL'].sum().reset_index()
      # plot_data(course_enrollment_summer)