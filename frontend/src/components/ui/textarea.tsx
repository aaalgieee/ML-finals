// File: src/components/ui/textarea.tsx
import React from 'react';

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  placeholder?: string;
}

export const Textarea: React.FC<TextareaProps> = ({ 
  value, 
  onChange, 
  placeholder,
  className = '',
  ...props 
}) => {
  return (
    <textarea
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      className={`
        w-full 
        px-4 
        py-3 
        bg-gray-50 
        rounded-md 
        focus:outline-none 
        focus:ring-2 
        focus:ring-blue-100 
        focus:bg-white 
        hover:bg-gray-100 
        transition-all 
        duration-200 
        min-h-[120px] 
        resize-y 
        text-gray-800 
        placeholder:text-gray-400
        ${className}
      `}
      {...props}
    />
  );
};