from game import Action
import numpy as np

ACTION_NUM = 356

card_dict = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6, '10': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12,
             'S': 13, 'B': 14}
r_card_dict = {}
for key in card_dict:
	r_card_dict[card_dict[key]] = key


def check_kicker(cards):
	total = cards.sum()
	for type_num in cards:
		if type_num < 3 and type_num != 0:
			for _type_num in cards:
				if _type_num >= 3:
					return False
		if type_num >= 3:
			if type_num == 4 and total == 4:
				return True  # boom
			if type_num == 3 and total % 3 == 0:
				return True
			return False
	return True


def to_action(_cards):
	cards = np.zeros(15, dtype=np.int)
	# print(_cards)
	for card in _cards:
		cards[card_dict[card]] += 1
	# print(cards)
	if check_kicker(cards):
		for action in range(ACTION_NUM):
			if Action(action).is_attach() or Action(action).need_attach():
				continue
			if (Action(action).to_array() == cards).all():
				return [action]
	else:
		suit = -1
		suit_num = 0
		for action in range(ACTION_NUM):
			action_consumed = Action(action).num() + Action(action).attach_num() * Action(action).attach_type()
			if Action(action).need_attach() and action_consumed == cards.sum():
				# print(Action(action).to_array())
				if ((cards - Action(action).to_array()) < 0).any():
					continue
				if Action(action).to_array().sum() > suit_num:
					suit_num = Action(action).to_array().sum()
					suit = action
		if suit != -1:
			ret = [suit]
			cards -= Action(suit).to_array()
			while cards.sum() > 0:
				flag = False
				for action in range(309, 337):
					if Action(action).num() == Action(suit).attach_type():
						if ((cards - Action(action).to_array()) >= 0).all():
							ret.append(action)
							cards -= Action(action).to_array()
							flag = True
							break
				if flag is False:
					break
			if cards.sum() == 0:
				return ret
	raise RuntimeError('WTF?')


def to_chars(_cards):
	cards = _cards.copy()
	ret = []
	for i in range(15):
		while cards[i]:
			cards[i] -= 1
			ret += [r_card_dict[i]]
	return str(ret).replace('\'', '')


def main():
	print(to_action(['A', 'A', 'A', '3', '3']))


if __name__ == '__main__':
	main()
