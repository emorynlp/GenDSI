from matryoshki.approach import Approach, DataView

if __name__ == '__main__':
    train_data = DataView('DST/mwoz24_train_hotel.pkl')[:3]
    train_data.save('DST/mwoz24_mini_train_hotel.pkl')
    valid_data = DataView('DST/mwoz24_test_hotel.pkl')[:3]
    valid_data.save('DST/mwoz24_mini_test_hotel.pkl')