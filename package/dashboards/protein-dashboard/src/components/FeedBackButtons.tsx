import React from 'react';
import { submitFeedback } from '../api';

interface FeedbackButtonsProps {
  queryId: string;
  onFeedbackSent: (feedback: 'positive' | 'negative') => void;
}

const FeedbackButtons: React.FC<FeedbackButtonsProps> = ({ queryId, onFeedbackSent }) => {
  const handleFeedback = async (positive: boolean) => {
    try {
      await submitFeedback(queryId, positive);
      onFeedbackSent(positive ? 'positive' : 'negative');
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  return (
    <div>
      <button onClick={() => handleFeedback(true)}>👍</button>
      <button onClick={() => handleFeedback(false)}>👎</button>
    </div>
  );
};

export default FeedbackButtons;